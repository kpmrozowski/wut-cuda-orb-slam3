#ifndef ORB_SLAM3_MANAGER_HPP
#define ORB_SLAM3_MANAGER_HPP
#include <chrono>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <functional>

#define MEASURE_METHOD_CALL(classname, func, ...) ::orb::benchmark::measure_function_void(::orb::benchmark::MeasuredFunction::func, &classname::func, this, __VA_ARGS__);
#define MEASURE_FUNC_CALL(func, ...) ::orb::benchmark::measure_function_void(::orb::benchmark::MeasuredFunction::func, func, __VA_ARGS__);

#define MEASURE_RET_METHOD_CALL(classname, func, ...) ::orb::benchmark::measure_function(::orb::benchmark::MeasuredFunction::func, &classname::func, this, __VA_ARGS__);
#define MEASURE_RET_CALL(func, ...) ::orb::benchmark::measure_function(::orb::benchmark::MeasuredFunction::func, func, __VA_ARGS__);

namespace orb::benchmark {

template<typename TValue>
class DBQueue {
  public:
    DBQueue() : m_front(&m_left),  m_back(&m_right) {}

    void push(TValue&& value) {
        std::unique_lock lock(m_write_mtx);
        m_back->push(std::forward<TValue>(value));
    }

    template<typename... TArgs>
    void emplace(TArgs&&... args) {
        std::unique_lock lock(m_write_mtx);
        m_back->push(TValue{args...});
    }

    std::optional<TValue> try_pop() {
        std::unique_lock lock(m_read_mtx);
        if (m_back->empty())
            return std::nullopt;

        auto value = m_back->front();
        m_back->pop();
        return std::move(value);
    }

    void swap_queues() {
        std::unique_lock lock1(m_read_mtx);
        std::unique_lock lock2(m_write_mtx);
        std::swap(m_front, m_back);
    }

  private:
    std::queue<TValue> m_left;
    std::queue<TValue> m_right;

    std::queue<TValue> *m_front{};
    std::queue<TValue> *m_back{};

    std::mutex m_read_mtx;
    std::mutex m_write_mtx;
};

enum class MeasuredFunction {
    Foo,
    Bar,
    DistributeOctTree,
    ComputeKeyPointsOctTree,
    ComputePyramid,
    computeDescriptors,
    GaussianBlur,
    GrabImageStereo,
    Track,
    computeOrientation,
    addBorderToCoordinates,
    runTileCalcKeypointsKernel_fun,
    extractorParenthesis,
    Action_Exit
};

constexpr std::string_view to_string(MeasuredFunction func) {
    switch (func) {
    case MeasuredFunction::Foo: return "foo";
    case MeasuredFunction::Bar: return "bar";
    case MeasuredFunction::Track: return "Track";
    case MeasuredFunction::GrabImageStereo: return "0.0.0_x1_GrabImageStereo";
    case MeasuredFunction::extractorParenthesis: return "1.0.0_x2_extractorParenthesis";
    case MeasuredFunction::ComputePyramid: return "1.1.0_x1_ComputePyramid";
    case MeasuredFunction::ComputeKeyPointsOctTree: return "1.2.0_x1_ComputeKeyPointsOctTree";
    case MeasuredFunction::runTileCalcKeypointsKernel_fun: return "1.2.1_x8_runTileCalcKeypointsKernel_fun";
    case MeasuredFunction::DistributeOctTree: return "1.2.2_x8_DistributeOctTree";
    case MeasuredFunction::addBorderToCoordinates: return "1.2.3_x8_addBorderToCoordinates";
    case MeasuredFunction::computeOrientation: return "1.2.4_x8_computeOrientation";
    case MeasuredFunction::GaussianBlur: return "1.3.0_x8_GaussianBlur";
    case MeasuredFunction::computeDescriptors: return "1.4.0_x8_computeDescriptors";
    default: break;
    }
    return "";
}

using Duration = std::chrono::nanoseconds;
using ClockCycles = int64_t;

struct LogEntry {
    MeasuredFunction function;
    Duration duration;
    Duration cycles;
};

enum class ExitStatus {
    Ok,
    CouldNotOpenFile
};

class Benchmark
{
  public:
    static Benchmark &the();

    void start(const std::string &filename);
    ExitStatus stop();
    ExitStatus write_routine(const std::string &filename);
    void log(MeasuredFunction func, Duration duration, Duration cycles);
  private:

    DBQueue<LogEntry> m_log_queue;
    std::future<ExitStatus> m_worker_thread;
};

template<typename TFunction, typename... TArgs>
inline auto measure_function(MeasuredFunction name, TFunction &&func, TArgs&& ...args) {
    auto before = std::chrono::system_clock::now();
    auto before_cycles = std::chrono::high_resolution_clock::now();
    auto&& result = std::invoke(func, std::forward<TArgs>(args)...);
    Benchmark::the().log(name, std::chrono::system_clock::now() - before,  std::chrono::high_resolution_clock::now() - before_cycles);
    return std::forward<std::decay_t<decltype(result)>>(result);
}

template<typename TFunction, typename... TArgs>
inline void measure_function_void(MeasuredFunction name, TFunction &&func, TArgs&& ...args) {
    auto before = std::chrono::system_clock::now();
    auto before_cycles = std::chrono::high_resolution_clock::now();
    std::invoke(func, std::forward<TArgs>(args)...);
    Benchmark::the().log(name, std::chrono::system_clock::now() - before,  std::chrono::high_resolution_clock::now() - before_cycles);
}

class BenchmarkInstance {
  public:
    explicit BenchmarkInstance(const std::string& filename) {
        Benchmark::the().start(filename);
    }

    ~BenchmarkInstance() {
        Benchmark::the().stop();
    }
};

}
#endif//ORB_SLAM3_MANAGER_HPP
