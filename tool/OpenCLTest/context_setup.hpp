#ifndef BOOST_COMPUTE_TEST_CONTEXT_SETUP_HPP
#define BOOST_COMPUTE_TEST_CONTEXT_SETUP_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>

#define REQUIRES_OPENCL_VERSION(major, minor) \
    if (!device.check_version(major, minor)) return

struct Context {
    boost::compute::device        device;
    boost::compute::context       context;
    boost::compute::command_queue queue;

    Context() :
        device ( boost::compute::system::default_device() ),
        context( boost::compute::system::default_context() ),
        queue  ( boost::compute::system::default_queue() )
    {}
};

BOOST_FIXTURE_TEST_SUITE(compute_test, Context)

#endif
