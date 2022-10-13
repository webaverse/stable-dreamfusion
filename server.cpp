#include <cstdio>
#include <iostream>
#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <array>
#include <queue>
#include <mutex>
#include <vector>
#include <filesystem>
#include <crow.h>
#include <ranges>
#include "threadpool.h"
#include <sstream>
#include <chrono>
#define CROW_MAIN

using std::string;
using std::mutex;
using std::lock_guard;
using std::make_shared;
using std::queue;
using std::vector;
namespace fs = std::filesystem;
namespace rv = std::ranges::views;

constexpr string to_st(const auto& i){ std::stringstream ss; ss << i; return ss.str(); }

static inline auto uid(const std::string& s){
	std::stringstream ss;
	std::hash<string> h;
	const auto t0 = std::chrono::system_clock::now();
	ss << s << '|' << std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count();
	return to_st(h(ss.str()));
}

static inline string exec(const char* cmd) {
	std::array<char, 128> buffer;
	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
	string result;
	if (!pipe)
		return "Command failed";
	else {
		while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
			result += buffer.data();
	}
	return result;
}

constexpr auto reify(const std::ranges::range auto& r){ return vector(r.begin(), r.end()); }

constexpr string strip(const string& s){ return s; }

constexpr string strip(string s, char ch){
	s.erase(std::remove_if(s.begin(), s.end(), [=](char c){ return c == ch; }), s.end());
	return s;
}

constexpr string strip(string s, char ch, auto... chs){
	return strip(strip(s, ch), chs...);
}

constexpr vector<string> splitOn(const string& s, const string& delim){
	vector<string> ret;
	long long int start = 0;
	for(size_t dp = s.find_first_of(delim, 0); start >= 0
	   ; start = (dp == std::string::npos) ? -1 : dp + 1, dp = s.find_first_of(delim, start))
		if(auto n = s.substr(start, dp - start); !n.empty())
			ret.emplace_back(n);
	return ret;
}

template <typename T>
constexpr auto q_to_v(queue<T> qcopy){
	vector<T> v;
	v.reserve(qcopy.size());
	while(!qcopy.empty())
		v.push_back(qcopy.front()), qcopy.pop();
	return v;
}

int main(){
	crow::SimpleApp app;
	typedef std::array<string, 2> guy;
	auto commissions = make_shared<queue<guy>>();
	auto queue_mutex = make_shared<mutex>()
	   , train_mutex = make_shared<mutex>();
	auto pool = make_shared<threadpool<>>(avail_threads() / 2);

	auto run = [=](const string& cmd){
		CROW_LOG_INFO << "running \'" << cmd;
		return exec(cmd.c_str());
	};

	auto poppe = [=](){
		lock_guard<mutex> qlock(*queue_mutex);
		commissions->pop();
		CROW_LOG_INFO << commissions->size() << " left in queue";
	};

	auto training_loop = [=](){
		lock_guard<mutex> lock(*train_mutex);
		while(!commissions->empty()){
			auto& [id, prompt] = commissions->front();
			CROW_LOG_INFO << "Launched training for prompt: " + prompt;
			run(string("sh train.sh \"") + prompt + "\"");
			CROW_LOG_INFO << run(string("sh upload.sh ") + id);
			CROW_LOG_INFO << "Finished training for prompt: " + prompt;
			poppe();
		}
	};

	auto enqueue = [=](const guy& thing){
		lock_guard<mutex> lock(*queue_mutex);
		commissions->push(thing);
		auto& [name, prompt] = thing;
		CROW_LOG_INFO << name << " queued with prompt: " << prompt;
	};

	CROW_ROUTE(app, "/create")
		.methods("GET"_method, "POST"_method)([=](const crow::request& req) -> string {
			if(auto prompt = req.url_params.get("prompt"); prompt == nullptr){
				CROW_LOG_INFO << "No prompt specified";
				return "Error: Can't train a NeRF without a prompt!";
			} else {
				CROW_LOG_INFO << prompt << " commissioned";
				auto id = uid(prompt);
				enqueue({id, strip(prompt, '\'', '\"')});
				pool->enqueue(training_loop);
				CROW_LOG_INFO << "Launched training loop";
				return "Scheduled training for " + id;
			}
		});

	CROW_ROUTE(app, "/list")([&](){
		std::vector<string> fin = splitOn(exec("ls *zip"), "\n")
						  , q = reify( q_to_v(*commissions) 
									 | rv::transform([](const guy& i){ return i[0] + ": '" + i[1] + "'"; }));
		crow::json::wvalue ret;
		ret["pending"] = q;
		return ret;
	});

	app.port(80).run();
}
