import { runScene } from '../../src/core/run';
import { LandscapeScene } from './scene';

runScene('#app', (ctx, frame, camera) => {
  camera.distance = 32;
  camera.pitch = 0.45;
  camera.far = 200;
  return new LandscapeScene(ctx, frame);
});
