typedef struct {
    float x;
    float y;
} Point2f;

typedef struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
} Point3b;

typedef struct {
    unsigned char *data;
    size_t step;
} PtrStepB;

typedef struct {
    int x;
    int y;
} Point2i;

typedef struct {
    Point2f pt;    //!< coordinates of the keyPoints
    float size;    //!< diameter of the meaningful keypoint neighborhood
    float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
    float response;//!< the response by which the most strong keyPoints have been selected. Can be used for the further sorting or subsampling
    int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
    int class_id;  //!< object class (if the keyPoints need to be clustered by an object they belong to)
} key_point_t;
