# Three.js browser bundle

`three-replay.min.js` bundles Three.js 0.180.0 and its OrbitControls add-on
from the official `three` npm package. Both are MIT licensed. See
`THREE-LICENSE.txt`.

The committed bundle lets the viewer load from both a local web server
and a directly opened HTML file without installing Node or fetching a CDN
script. To rebuild, install `three@0.180.0` and `esbuild@0.25.10` in a
temporary Node project and save the following as `entry.js` there:

```js
export * from "three";
export { OrbitControls } from "three/addons/controls/OrbitControls.js";
```

```text
esbuild entry.js --bundle --minify --format=iife --global-name=ThreeReplay --target=es2020 --outfile=three-replay.min.js
```
