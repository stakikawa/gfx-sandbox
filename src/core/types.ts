import type { Mat4, Vec3 } from 'wgpu-matrix';

// Immutable GPU handles every layer shares.
export interface GpuContext {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  presentFormat: GPUTextureFormat; // 8-bit swapchain format, for the final present
  hdrFormat: GPUTextureFormat; // rgba16float scene color target
  depthFormat: GPUTextureFormat; // shared scene depth target
}

// Per-frame CPU state, mirrored into the group-0 uniform buffer each frame.
export interface FrameState {
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  invView: Mat4;
  invProj: Mat4;
  cameraPos: Vec3;
  time: number;
  resolution: [number, number];
  sunDir: Vec3;
  sunIntensity: number;
  sunViewProj: Mat4; // orthographic light view-projection for the shadow map
}

// Render targets core hands the scene each frame.
export interface FrameTargets {
  color: GPUTextureView; // HDR target the scene draws into
  depth: GPUTextureView; // shared scene depth
}

// A scene owns composition: it encodes its passes in a deliberate order.
export interface Scene {
  encode(encoder: GPUCommandEncoder, frame: FrameState, targets: FrameTargets): void;
}
