// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeSpheresNaiveCpu(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int points_per_pixel);

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSpheresNaiveCuda(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int points_per_pixel);
#endif
// Naive (forward) pointcloud rasterization: For each pixel, for each point,
// check whether that point hits the pixel.
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of each point (in NDC units)
//  image_size: (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number closest of points to return for each pixel
//
// Returns:
//  A 4 element tuple of:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each
//        closest point for each pixel.
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//          distance in the (NDC) x/y plane between each pixel and its K closest
//          points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeSpheresNaive(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int points_per_pixel) {
  if (spheres.is_cuda() && cloud_to_packed_first_idx.is_cuda() &&
      num_points_per_cloud.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(spheres);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    return RasterizeSpheresNaiveCuda(
        spheres,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        points_per_pixel);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeSpheresNaiveCpu(
        spheres,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        points_per_pixel);
  }
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

torch::Tensor RasterizeSpheresCoarseCpu(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int bin_size,
    const int max_points_per_bin);

#ifdef WITH_CUDA
torch::Tensor RasterizeSpheresCoarseCuda(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int bin_size,
    const int max_points_per_bin);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of points to rasterize (in NDC units)
//  image_size: Size of the image to generate (in pixels)
//  bin_size: Size of each bin within the image (in pixels)
//
// Returns:
//  points_per_bin: Tensor of shape (N, num_bins, num_bins) giving the number
//                  of points that fall in each bin
//  bin_points: Tensor of shape (N, num_bins, num_bins, K) giving the indices
//              of points that fall into each bin.
torch::Tensor RasterizeSpheresCoarse(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int bin_size,
    const int max_points_per_bin) {
  if (spheres.is_cuda() && cloud_to_packed_first_idx.is_cuda() &&
      num_points_per_cloud.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(spheres);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    return RasterizeSpheresCoarseCuda(
        spheres,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        bin_size,
        max_points_per_bin);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeSpheresCoarseCpu(
        spheres,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        bin_size,
        max_points_per_bin);
  }
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeSpheresFineCuda(
    const torch::Tensor& spheres,
    const torch::Tensor& bin_points,
    const int image_size,
    const int bin_size,
    const int points_per_pixel);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  bin_points: int32 Tensor of shape (N, B, B, M) giving the indices of points
//              that fall into each bin (output from coarse rasterization)
//  image_size: Size of image to generate (in pixels)
//  radius: Radius of points to rasterize (NDC units)
//  bin_size: Size of each bin (in pixels)
//  points_per_pixel: How many points to rasterize for each pixel
//
// Returns (same as rasterize_points):
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//         distance in the (NDC) x/y plane between each pixel and its K closest
//         points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeSpheresFine(
    const torch::Tensor& spheres,
    const torch::Tensor& bin_points,
    const int image_size,
    const int bin_size,
    const int points_per_pixel) {
  if (spheres.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(spheres);
    CHECK_CUDA(bin_points);
    return RasterizeSpheresFineCuda(
        spheres, bin_points, image_size, bin_size, points_per_pixel);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("NOT IMPLEMENTED");
  }
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

torch::Tensor RasterizeSpheresBackwardCpu(
    const torch::Tensor& spheres,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists);

#ifdef WITH_CUDA
torch::Tensor RasterizeSpheresBackwardCuda(
    const torch::Tensor& spheres,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  idxs: int32 Tensor of shape (N, H, W, K) (from forward pass)
//  grad_zbuf: float32 Tensor of shape (N, H, W, K) giving upstream gradient
//             d(loss)/d(zbuf) of the distances from each pixel to its nearest
//             points.
//  grad_dists: Tensor of shape (N, H, W, K) giving upstream gradient
//              d(loss)/d(dists) of the dists tensor returned by the forward
//              pass.
//
// Returns:
//  grad_points: float32 Tensor of shape (N, P, 3) giving downstream gradients
torch::Tensor RasterizeSpheresBackward(
    const torch::Tensor& spheres,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists) {
  if (spheres.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(spheres);
    CHECK_CUDA(idxs);
    CHECK_CUDA(grad_zbuf);
    CHECK_CUDA(grad_dists);
    return RasterizeSpheresBackwardCuda(spheres, idxs, grad_zbuf, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeSpheresBackwardCpu(spheres, idxs, grad_zbuf, grad_dists);
  }
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************

// This is the main entry point for the forward pass of the point rasterizer;
// it uses either naive or coarse-to-fine rasterization based on bin_size.
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of each point (in NDC units)
//  image_size:  (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number of points to return for each pixel
//  bin_size: Bin size (in pixels) for coarse-to-fine rasterization. Setting
//            bin_size=0 uses naive rasterization instead.
//  max_points_per_bin: The maximum number of points allowed to fall into each
//                      bin when using coarse-to-fine rasterization.
//
// Returns:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//         distance in the (NDC) x/y plane between each pixel and its K closest
//         points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeSpheres(
    const torch::Tensor& spheres,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const int points_per_pixel,
    const int bin_size,
    const int max_points_per_bin) {
  if (bin_size == 0) {
    // Use the naive per-pixel implementation
    return RasterizeSpheresNaive(
        spheres,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        points_per_pixel);
  } else {
    // Use coarse-to-fine rasterization
    const auto bin_points = RasterizeSpheresCoarse(
        spheres,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        bin_size,
        max_points_per_bin);
    return RasterizeSpheresFine(
        spheres, bin_points, image_size, bin_size, points_per_pixel);
  }
}
