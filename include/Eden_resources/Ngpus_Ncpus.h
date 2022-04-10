#ifndef NGPUS_NCPUS_H
#define NGPUS_NCPUS_H

class Eden_resources {
public:
  static unsigned get_gpus_count();
  static unsigned get_cpus_count();
};

#endif //NGPUS_NCPUS_H