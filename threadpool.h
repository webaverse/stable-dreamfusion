#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>
#include <concepts>
#include <thread>
static inline size_t avail_threads(){
	return std::thread::hardware_concurrency();
}

template <typename F, typename... Args>
using irt = std::invoke_result_t<F, Args...>;

template <typename F, typename... Args>
static inline auto taskify(const F& funk, Args... args){
	return std::bind(funk, std::forward<Args...>(args)...);
}

static inline auto promitask(const std::invocable<> auto& funk){
	typedef irt<decltype(funk)> ret;
	auto pr = std::make_shared<std::promise<ret>>();
	if constexpr (std::same_as<irt<decltype(funk)>, void>)
		return make_pair([=](){ funk(), pr->set_value(); }, pr);
	else
		return make_pair([=](){ pr->set_value(funk()); }, pr);
}

static inline auto await_many(const std::ranges::range auto& fu){
	std::for_each(fu.begin(), fu.end(), [](auto&& f){ f.wait(); });
}

typedef std::function<void()> tasque;

template <typename Thrd = std::jthread>
class threadpool {
	/// If true the queue thread should exit
	std::atomic<bool> done;

	/// The thread object associated with this queue
	std::vector<Thrd> queue_threads;
	/// A queue of functions that will be executed on the queue thread
	std::queue<tasque> work_queue;

	/// The mutex used in the condition variable
	std::mutex queue_mutex;

	/// The condition variable that waits for a new function to be inserted in the
	/// queue
	std::condition_variable cond;

	/// This funciton executes on the queue_thread
	void queue_runner() {
		while (!done) {
			tasque func;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				cond.wait( lock
						 , [this]() { return work_queue.empty() == false || done; });

				if (!done){
					swap(func, work_queue.front());
					work_queue.pop();
				}
			}
			if (func) func();
		}
	}

	void qup(const std::invocable<> auto& f){
		std::lock_guard<std::mutex> lock(queue_mutex);
		work_queue.push(f);
		cond.notify_one();
	}

public:
	template <typename F, typename... Args>
	void enqueue(const F& func, Args... args) requires std::invocable<F, Args...> {
		qup(taskify(func, args...));
	}

	template <typename F, typename... Args>
	auto inquire(const F& func, Args... args) requires std::invocable<F, Args...> {
		auto [t, pr] = promitask(taskify(func, args...));
		auto fut = pr->get_future();
		enqueue(t);
		return fut;
	}

	void clear() {
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			while(!work_queue.empty())
				work_queue.pop();
		}
		sync();
	}

	void sync(){
		std::atomic<size_t> n(0);
		const size_t m = queue_threads.size();
		auto present = [&](){ ++n; size_t l = n.load(); while(l < m) l = n.load(); };
		std::vector<std::future<void>> fu;
		std::ranges::generate_n(std::back_inserter(fu), m, [=, this](){ return inquire(present); });
		await_many(fu);
	}

	threadpool(size_t n, size_t res) : done(false)
									 , queue_threads(n ? std::clamp(n, size_t(1), avail_threads() - res)
													   : std::max(size_t(1), avail_threads() - res)) {
		for(auto& i:queue_threads){
			Thrd tmp(&threadpool::queue_runner, this);
			std::swap(i, tmp);
		}
	}
	threadpool(size_t n) : threadpool(n, 0) {}
	threadpool() : threadpool(0, 1) {}

	~threadpool() {
		sync();
		done.store(true);
		cond.notify_all();
	}

	threadpool(const threadpool& other) : work_queue(other.work_queue), done(false) {
		for(auto& i:queue_threads){
			Thrd tmp(&threadpool::queue_runner, this);
			std::swap(i, tmp);
		}
	}

	threadpool& operator=(const threadpool& other){
		clear();
		work_queue = other.work_queue;
		return *this;
	}
	size_t size() const { return queue_threads.size(); }
	threadpool& operator=(threadpool&& other) = default;
	threadpool(threadpool&& other) = default;
};

