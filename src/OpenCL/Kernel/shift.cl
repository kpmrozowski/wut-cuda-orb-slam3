

__constant sampler_t samplerLN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__kernel void shift(
   const image2d_t src,
   float shift_x,
   float shift_y,
   __global uchar* dst,
   int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   if (x >= dst_cols) return;
   int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT), dst_offset));
   __global dstT *dstf = (__global dstT *)(dst + dst_index);
   float2 coord = (float2)((float)x+0.5f+shift_x, (float)y+0.5f+shift_y);
   dstf[0] = (dstT)read_imagef(src, samplerLN, coord).x;
}
