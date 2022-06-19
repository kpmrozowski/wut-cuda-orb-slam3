__kernel void squareVector(__global int *data) {
    const int globalId   = get_global_id(0);
    data[globalId] = work_group_reduce_add(data[globalId]);
    data[globalId] = data[globalId]+1;
}
