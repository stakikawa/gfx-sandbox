import type { Mat4, Vec3 } from 'wgpu-matrix';

// Immutable GPU handles every layer shares.
export interface GpuContext {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  presentFormat: GPUTextureFormat; // 8-bit swapchain format, for the final present
  hdrFormat: GPUTextureFormat; // rgba16float scene color target
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
}

// Render targets core hands the scene each frame.
export interface FrameTargets {
  color: GPUTextureView; // HDR target the scene draws into
}

// A scene owns composition: it encodes its passes in a deliberate order.
export interface Scene {
  encode(encoder: GPUCommandEncoder, frame: FrameState, targets: FrameTargets): void;
}
