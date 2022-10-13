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
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <sstream>
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

static inline auto uid(const std::string& s){ std::hash<string> h; return to_st(h(s)); }

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


static inline string train(const string& prompt){
	return string("python main.py --save_mesh --text \"") + strip(prompt, '\'', '\"') + "\" --workspace trial -O";
}

template <typename T>
constexpr auto q_to_v(queue<T> qcopy){
	vector<T> v;
	v.reserve(qcopy.size());
	while(!qcopy.empty())
		v.push_back(qcopy.front()), qcopy.pop();
	return v;
}

constexpr auto stdfopts = std::ios_base::in | std::ios_base::binary;
static inline auto awssave(){
	auto options = make_shared<Aws::SDKOptions>();
	Aws::InitAPI(*options);
	Aws::Client::ClientConfiguration clientConfig;
	std::shared_ptr<Aws::S3::S3Client> s3_client( new Aws::S3::S3Client(clientConfig)
									  , [=](Aws::S3::S3Client* c){ 
											Aws::ShutdownAPI(*options);
											delete c;
									 });
	return [=](const std::string& id){
		Aws::S3::Model::PutObjectRequest request;
		request.SetBucket("models.webaverse.com");
		request.SetKey(id);
		if(auto in = Aws::MakeShared<Aws::FStream>("a", "model.glb", stdfopts); *in){
			request.SetBody(in);
			auto outcome = s3_client->PutObject(request);
			if(outcome.IsSuccess())
				std::cout << "Added '" << id << "' to bucket 'models.webaverse.com'\n";
			else
				std::cerr << "AWS S3 Error: " << outcome.GetError().GetMessage() << "\n";
			return outcome.IsSuccess();
		} else {
			std::cerr << "No such file: model.glb\n";
			return false;
		}
	};
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

	auto check_queue = [=](const string& name) -> int {
		if(!commissions->empty()){
			const auto v = q_to_v(*commissions);
			if(auto pos = std::find_if( v.begin(), v.end()
									  , [=](const guy& g){ auto& [n,d] = g; return n == name; });
			pos != v.end())
				return int(std::distance(v.begin(), pos));
		}
		return -1;
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
			run("rm -rf trial/checkpoints/*");
			run(train(prompt));
			CROW_LOG_INFO << run("aws s3 cp model.glb s3://models.webaverse.com/" + id + ".glb");
			run("rm model.glb");
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

	CROW_ROUTE(app, "/create/<string>")
		.methods("GET"_method, "POST"_method)([=](const crow::request& req, const string& prompt) -> string {
			CROW_LOG_INFO << prompt << " commissioned";
			if(auto id = uid(prompt); prompt.empty()){
				CROW_LOG_INFO << "No prompt specified";
				return "Error: Can't train a NeRF without a prompt!";
			} else {
				if(auto r = check_queue(id); r < 0){
					enqueue({id, prompt});
					pool->enqueue(training_loop);
					CROW_LOG_INFO << "Launched training loop";
					return "Scheduled training for " + id;
				} else
					return id + " is currently " + (r ? string("in line") : string("training"));
			}
		});

	CROW_ROUTE(app, "/check/<string>")([=](crow::response& res, const string& name){
		CROW_LOG_INFO << name << " check'd";
		if(fs::exists(fs::path(name + ".zip"))){
			res.write("O I know that guy");
			res.set_static_file_info(name + ".zip");
		} else if(auto r = check_queue(name); r < 0)
			res.write("Doesn't look like much of anything to me");
		else
			res.write(name + " is currently "
						   + (r ? string("in line") : string("training")));
		res.end();
	});

	CROW_ROUTE(app, "/list")([&](){
		std::vector<string> fin = splitOn(exec("ls *zip"), "\n")
						  , q = reify(q_to_v(*commissions) | rv::transform([](const guy& i){ return i[0]; }));
		crow::json::wvalue ret;
		ret["finished"] = fin;
		ret["pending"] = q;
		return ret;
	});

	app.port(80).run();
}
