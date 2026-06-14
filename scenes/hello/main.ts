import { runScene } from '../../src/core/run';
import { HelloScene } from './scene';

runScene('#app', (ctx, frame) => new HelloScene(ctx, frame));
