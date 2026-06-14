import type { FrameState } from './types';

// WGSL definition of the group-0 frame uniform. Kept next to the byte-packing in update()
// so the two never drift; shaders prepend this string.
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  viewProj : mat4x4<f32>,
  invView : mat4x4<f32>,
  invProj : mat4x4<f32>,
  cameraPos : vec3<f32>,
  time : f32,
  resolution : vec2<f32>,
  sunIntensity : f32,
  _pad0 : f32,
  sunDir : vec3<f32>,
  _pad1 : f32,
  sunViewProj : mat4x4<f32>,
};
`;

const FLOATS = 108; // 432 bytes, std140 layout matching FRAME_WGSL

// The shared group-0 frame uniform buffer plus its bind group.
export class Frame {
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  private readonly buffer: GPUBuffer;
  private readonly data = new Float32Array(FLOATS);

  constructor(private readonly device: GPUDevice) {
    this.buffer = device.createBuffer({
      label: 'frame-ubo',
      size: FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'frame',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
      ],
    });
    this.bindGroup = device.createBindGroup({
      label: 'frame',
      layout: this.bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.buffer } }],
    });
  }

  update(s: FrameState): void {
    const d = this.data;
    d.set(s.view, 0);
    d.set(s.proj, 16);
    d.set(s.viewProj, 32);
    d.set(s.invView, 48);
    d.set(s.invProj, 64);
    d.set(s.cameraPos, 80);
    d[83] = s.time;
    d[84] = s.resolution[0];
    d[85] = s.resolution[1];
    d[86] = s.sunIntensity;
    d.set(s.sunDir, 88);
    d.set(s.sunViewProj, 92);
    this.device.queue.writeBuffer(this.buffer, 0, d);
  }
}
