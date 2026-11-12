// All positions here are display geometry in the replay's X-Z plane.
const PAYLOAD_DEPTH = 0.8; // m, visual only
const MIN_DIRECTION_FORCE = 0.025; // N, visual only
const UAV_COLORS = [0x008477, 0xc66b29, 0x616bc4, 0xb34e8a, 0x6f8b25, 0x197eac];

class ReplayScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new ThreeReplay.WebGLRenderer({canvas, antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = ThreeReplay.PCFShadowMap;
    this.scene = new ThreeReplay.Scene();
    this.scene.background = new ThreeReplay.Color(0xe9eff1);
    this.scene.add(new ThreeReplay.HemisphereLight(0xf6fbff, 0x718189, 2.15));
    this.sun = new ThreeReplay.DirectionalLight(0xfff6e9, 2.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    this.sun.shadow.normalBias = 0.035;
    this.scene.add(this.sun, this.sun.target);
    this.world = new ThreeReplay.Group();
    this.scene.add(this.world);
    this.center = new ThreeReplay.Vector3();
    this.lastFollow = new ThreeReplay.Vector3();
    this.scratch = new ThreeReplay.Vector3();
    this.inverseCamera = new ThreeReplay.Quaternion();
    this.axesCanvas = document.getElementById("camera-axes");
    this.axesContext = this.axesCanvas.getContext("2d");
    this.view = "perspective";
    this.follow = false;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
  }

  mesh(geometry, material, parent, x = 0, y = 0, z = 0) {
    const mesh = new ThreeReplay.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }

  label(text, color, size) {
    const bitmap = document.createElement("canvas");
    bitmap.width = 192;
    bitmap.height = 64;
    const ctx = bitmap.getContext("2d");
    ctx.font = "600 38px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = "#f5fafc";
    ctx.strokeText(text, 96, 33);
    ctx.fillStyle = color;
    ctx.fillText(text, 96, 33);
    const map = new ThreeReplay.CanvasTexture(bitmap);
    map.colorSpace = ThreeReplay.SRGBColorSpace;
    const sprite = new ThreeReplay.Sprite(new ThreeReplay.SpriteMaterial({
      map, transparent: true, depthWrite: false,
    }));
    sprite.scale.set(size, size / 3, 1);
    return sprite;
  }

  load(replay) {
    const geometries = new Set(), materials = new Set();
    this.world.traverse((object) => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) materials.add(object.material);
    });
    for (const geometry of geometries) geometry.dispose();
    for (const material of materials) {
      material.map?.dispose();
      material.dispose();
    }
    this.world.clear();
    this.replay = replay;
    const {width, height, attachments} = replay.payload;
    const solid = new ThreeReplay.MeshStandardMaterial({color: 0x245563, roughness: 0.62});
    const trim = new ThreeReplay.MeshStandardMaterial({color: 0x678d98, roughness: 0.43, metalness: 0.24});
    const box = new ThreeReplay.BoxGeometry(width, PAYLOAD_DEPTH, height);
    this.payload = new ThreeReplay.Group();
    this.world.add(this.payload);
    this.mesh(box, solid, this.payload);
    for (const x of [-width * 0.37, width * 0.37]) {
      this.mesh(new ThreeReplay.BoxGeometry(width * 0.055, PAYLOAD_DEPTH + 0.025, height + 0.025), trim, this.payload, x);
    }
    const marker = new ThreeReplay.SphereGeometry(Math.min(width, height) * 0.047, 16, 10);
    const centerMaterial = new ThreeReplay.MeshBasicMaterial({color: 0xfff1bd, depthTest: false});
    const com = this.mesh(marker, centerMaterial, this.payload);
    com.castShadow = false;
    com.renderOrder = 3;
    const centerRing = this.mesh(new ThreeReplay.RingGeometry(height * 0.07, height * 0.085, 28),
      new ThreeReplay.MeshBasicMaterial({color: 0xfff1bd, side: ThreeReplay.DoubleSide, depthTest: false}),
      this.payload);
    centerRing.rotation.x = Math.PI / 2;
    centerRing.castShadow = false;
    centerRing.renderOrder = 3;

    this.ghost = new ThreeReplay.Mesh(box, new ThreeReplay.MeshBasicMaterial({
      color: 0xf3ac49, opacity: 0.14, transparent: true, depthWrite: false,
    }));
    this.ghost.add(new ThreeReplay.LineSegments(new ThreeReplay.EdgesGeometry(box),
      new ThreeReplay.LineBasicMaterial({color: 0xd98927, transparent: true, opacity: 0.7})));
    this.world.add(this.ghost);

    this.drones = attachments.map(([x, z], i) => {
      const tint = UAV_COLORS[i % UAV_COLORS.length];
      const markerMaterial = new ThreeReplay.MeshBasicMaterial({color: tint, depthTest: false});
      const attachment = this.mesh(marker, markerMaterial, this.payload, x, 0, z);
      attachment.castShadow = false;
      attachment.renderOrder = 3;
      return this.quadrotor(tint, i);
    });

    const lastDirection = attachments.map(() => Math.PI / 2);
    // Store fallback directions so seeking backwards gives the same picture.
    this.directions = replay.u.map((row) => lastDirection.map((angle, i) => {
      const fx = row[2 * i], fz = row[2 * i + 1];
      if (Math.hypot(fx, fz) <= MIN_DIRECTION_FORCE) return angle;
      const raw = Math.atan2(fz, fx);
      const delta = Math.atan2(Math.sin(raw - angle), Math.cos(raw - angle));
      lastDirection[i] = angle + delta;
      return lastDirection[i];
    }));

    const bounds = new ThreeReplay.Box3();
    const point = new ThreeReplay.Vector3();
    for (const history of [replay.state, replay.reference]) {
      for (const [x, z] of history) bounds.expandByPoint(point.set(x, 0, z));
    }
    let reach = Math.hypot(width, height) / 2;
    for (const [x, z] of attachments) reach = Math.max(reach, Math.hypot(x, z));
    bounds.min.addScalar(-reach);
    bounds.max.addScalar(reach);
    bounds.max.z += 2.3;
    bounds.getCenter(this.center);
    this.span = Math.max(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z, 4);
    this.makeEnvironment(bounds.min.z - 0.32);
    this.actual = this.path(replay.state, 0x0a877b);
    this.reference = this.path(replay.reference, 0xd98927, true);
    this.trailTip = this.path([[0, 0], [0, 0]], 0x0a877b);
    this.trailTip.geometry.attributes.position.setUsage(ThreeReplay.DynamicDrawUsage);
    this.trailTip.frustumCulled = false;
    this.setView(this.view);
  }

  quadrotor(tint, index) {
    const {Group, MeshStandardMaterial, BoxGeometry, CylinderGeometry, SphereGeometry} = ThreeReplay;
    const model = new Group();
    this.world.add(model);
    const carbon = new MeshStandardMaterial({color: 0x222c32, roughness: 0.65, metalness: 0.23});
    const shell = new MeshStandardMaterial({color: 0xcbd7da, roughness: 0.44, metalness: 0.19});
    const accent = new MeshStandardMaterial({color: tint, roughness: 0.42});
    const rotor = new MeshStandardMaterial({color: 0x485c66, roughness: 0.56});
    const body = this.mesh(new SphereGeometry(1, 16, 10), shell, model, 0, 0, 0.06);
    body.scale.set(0.24, 0.18, 0.096);
    this.mesh(new BoxGeometry(0.36, 0.24, 0.10), carbon, model);
    this.mesh(new BoxGeometry(0.15, 0.22, 0.033), accent, model, 0, -0.04, 0.15);
    this.mesh(new BoxGeometry(0.10, 0.09, 0.06), carbon, model, 0, -0.17, 0.026);
    const props = [];
    const armLength = 0.39;
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        const x = sx * 0.28, y = sy * 0.28;
        const arm = this.mesh(new BoxGeometry(armLength, 0.055, 0.045), carbon, model, x / 2, y / 2, 0.02);
        arm.rotation.z = Math.atan2(y, x);
        const motor = this.mesh(new CylinderGeometry(0.064, 0.071, 0.08, 12), accent, model, x, y, 0.065);
        motor.rotation.x = Math.PI / 2;
        const propeller = new Group();
        propeller.position.set(x, y, 0.12);
        model.add(propeller);
        for (const side of [-1, 1]) {
          const blade = this.mesh(new SphereGeometry(1, 10, 6), rotor, propeller, side * 0.072, side * 0.014, 0);
          blade.scale.set(0.080, 0.022, 0.009);
          blade.rotation.z = 0.18;
        }
        const hub = this.mesh(new CylinderGeometry(0.026, 0.026, 0.026, 10), carbon, propeller);
        hub.rotation.x = Math.PI / 2;
        props.push(propeller);
      }
    }
    for (const x of [-0.15, 0.15]) {
      this.mesh(new BoxGeometry(0.037, 0.40, 0.033), carbon, model, x, 0, -0.15);
      for (const y of [-0.12, 0.12]) {
        this.mesh(new BoxGeometry(0.032, 0.032, 0.12), carbon, model, x, y, -0.086);
      }
    }
    const label = this.label(`UAV ${index + 1}`, "#223844", 1.12);
    label.position.z = index % 2 ? 0.57 : 0.32;
    model.add(label);
    const cable = new ThreeReplay.Line(
      new ThreeReplay.BufferGeometry().setAttribute("position", new ThreeReplay.BufferAttribute(new Float32Array(6), 3)),
      new ThreeReplay.LineBasicMaterial({color: tint}),
    );
    cable.geometry.attributes.position.setUsage(ThreeReplay.DynamicDrawUsage);
    cable.frustumCulled = false;
    const arrow = new ThreeReplay.ArrowHelper(new ThreeReplay.Vector3(0, 0, 1), new ThreeReplay.Vector3(), 1, tint);
    this.world.add(cable, arrow);
    return {model, props, label, cable, arrow};
  }

  makeEnvironment(groundZ) {
    const size = Math.ceil(Math.max(18, this.span * 2.7) / 2) * 2;
    const floor = this.mesh(new ThreeReplay.PlaneGeometry(size, size),
      new ThreeReplay.MeshStandardMaterial({color: 0xe3ebed, roughness: 1}),
      this.world, this.center.x, 0, groundZ);
    floor.castShadow = false;
    const grid = new ThreeReplay.GridHelper(size, size, 0x94a7b0, 0xbccbd1);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(this.center.x, 0, groundZ + 0.005);
    this.world.add(grid);
    const origin = new ThreeReplay.Vector3();
    const axes = [
      ["X", new ThreeReplay.Vector3(1, 0, 0), "#ac4240"],
      ["Y", new ThreeReplay.Vector3(0, 1, 0), "#4265b4"],
      ["Z", new ThreeReplay.Vector3(0, 0, 1), "#277b50"],
    ];
    for (const [text, direction, color] of axes) {
      const axisOrigin = origin.clone().addScaledVector(direction, -0.20);
      const axis = new ThreeReplay.ArrowHelper(direction, axisOrigin, 1.25, color, 0.14, 0.08);
      const label = this.label(text, color, 0.67);
      label.position.copy(direction).multiplyScalar(1.23);
      this.world.add(axis, label);
    }

    const span = this.span * 0.9;
    this.sun.position.copy(this.center).add(new ThreeReplay.Vector3(-span * 0.6, -span * 0.55, span * 1.65));
    this.sun.target.position.copy(this.center);
    Object.assign(this.sun.shadow.camera, {left: -span, right: span, top: span, bottom: -span, near: 0.1, far: span * 5});
    this.sun.shadow.camera.updateProjectionMatrix();
  }

  path(states, color, dashed = false) {
    const positions = new Float32Array(states.length * 3);
    states.forEach(([x, z], i) => { positions[3 * i] = x; positions[3 * i + 2] = z; });
    const geometry = new ThreeReplay.BufferGeometry().setAttribute("position", new ThreeReplay.BufferAttribute(positions, 3));
    const material = dashed
      ? new ThreeReplay.LineDashedMaterial({color, dashSize: 0.14, gapSize: 0.10})
      : new ThreeReplay.LineBasicMaterial({color});
    const line = new ThreeReplay.Line(geometry, material);
    if (dashed) line.computeLineDistances();
    this.world.add(line);
    return line;
  }

  setView(view) {
    this.view = view;
    if (!this.replay) return;
    this.controls?.dispose();
    const perspective = view === "perspective";
    this.camera = perspective
      ? new ThreeReplay.PerspectiveCamera(38, 1, 0.04, this.span * 30)
      : new ThreeReplay.OrthographicCamera(-1, 1, 1, -1, 0.04, this.span * 30);
    // The top view looks along -Z, so +Y defines the screen's vertical axis there.
    this.camera.up.set(0, view === "top" ? 1 : 0, view === "top" ? 0 : 1);
    const direction = {
      perspective: [0.62, -1, 0.62],
      front: [0, -1, 0],
      side: [1, 0, 0],
      top: [0, 0, 1],
    }[view];
    const target = this.follow ? this.payload.position : this.center;
    const distance = this.span * (perspective ? 1.42 : 2);
    this.camera.position.copy(target).addScaledVector(new ThreeReplay.Vector3(...direction).normalize(), distance);
    this.controls = new ThreeReplay.OrbitControls(this.camera, this.canvas);
    this.controls.target.copy(target);
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = this.span * 8;
    this.controls.minZoom = 0.2;
    this.controls.maxZoom = 12;
    this.controls.update();
    this.controls.addEventListener("change", () => this.render());
    this.lastFollow.copy(this.payload.position);
    this.resize();
  }

  setFollow(follow) {
    this.follow = follow;
    if (this.replay) this.setView(this.view);
  }

  resize() {
    const {width, height} = this.canvas.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    if (!this.camera) return;
    const aspect = width / height;
    if (this.camera.isPerspectiveCamera) {
      this.camera.aspect = aspect;
      this.camera.fov = ThreeReplay.MathUtils.radToDeg(2 * Math.atan(Math.tan(38 * Math.PI / 360) / Math.min(aspect, 1)));
    } else {
      const half = this.span * 0.56 / Math.min(aspect, 1);
      Object.assign(this.camera, {left: -half * aspect, right: half * aspect, top: half, bottom: -half});
    }
    this.camera.updateProjectionMatrix();
    this.render();
  }

  update({pose, reference, forces, index, fraction, time, visible, cableLength, arrowScale}) {
    const [x, z, theta] = pose;
    this.payload.position.set(x, 0, z);
    this.payload.rotation.y = -theta;
    this.ghost.position.set(reference[0], 0, reference[1]);
    this.ghost.rotation.y = -reference[2];
    this.ghost.visible = this.reference.visible = visible.reference;
    this.actual.visible = this.trailTip.visible = visible.actual;
    this.actual.geometry.setDrawRange(0, index + 1);
    const [sampleX, sampleZ] = this.replay.state[index];
    const tip = this.trailTip.geometry.attributes.position;
    tip.setXYZ(0, sampleX, 0, sampleZ);
    tip.setXYZ(1, x, 0, z);
    tip.needsUpdate = true;

    const cos = Math.cos(theta), sin = Math.sin(theta);
    const direction = this.scratch;
    const next = Math.min(index + 1, this.directions.length - 1);
    this.replay.payload.attachments.forEach(([bx, bz], i) => {
      const {model, props, label, cable, arrow} = this.drones[i];
      const ax = x + cos * bx - sin * bz;
      const az = z + sin * bx + cos * bz;
      const fx = forces[2 * i], fz = forces[2 * i + 1];
      const norm = Math.hypot(fx, fz);
      const held = this.directions[index][i];
      const fallback = held + fraction * (this.directions[next][i] - held);
      let angle = fallback;
      if (norm > MIN_DIRECTION_FORCE) {
        const raw = Math.atan2(fz, fx);
        const delta = Math.atan2(Math.sin(raw - fallback), Math.cos(raw - fallback));
        const blend = Math.min(1, (norm - MIN_DIRECTION_FORCE) / MIN_DIRECTION_FORCE);
        angle += blend * delta;
      }
      model.position.set(ax + cableLength * Math.cos(angle), 0, az + cableLength * Math.sin(angle));
      label.visible = visible.labels;
      props.forEach((prop, j) => { prop.rotation.z = (j % 2 ? -1 : 1) * time * 38 + j; });
      const points = cable.geometry.attributes.position;
      points.setXYZ(0, ax, 0, az);
      points.setXYZ(1, model.position.x, 0, model.position.z - 0.15);
      points.needsUpdate = true;
      cable.visible = visible.cables;
      arrow.visible = visible.forces && norm > 1e-8;
      if (arrow.visible) {
        arrow.position.set(ax, 0, az);
        direction.set(fx / norm, 0, fz / norm);
        arrow.setDirection(direction);
        const length = arrowScale * norm;
        const head = Math.min(0.19, length * 0.38);
        arrow.setLength(length, head, head * 0.68);
      }
    });
    if (this.follow) {
      const delta = direction.copy(this.payload.position).sub(this.lastFollow);
      this.camera.position.add(delta);
      this.controls.target.add(delta);
      this.lastFollow.copy(this.payload.position);
    }
    this.render();
  }

  render() {
    if (!this.camera) return;
    this.renderer.render(this.scene, this.camera);
    this.drawAxes();
  }

  drawAxes() {
    const ctx = this.axesContext;
    const center = this.axesCanvas.width / 2;
    ctx.clearRect(0, 0, this.axesCanvas.width, this.axesCanvas.height);
    ctx.beginPath();
    ctx.arc(center, center, 64, 0, Math.PI * 2);
    ctx.fillStyle = "#f7fafbe8";
    ctx.fill();
    ctx.font = "600 24px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 4;
    const inverse = this.inverseCamera.copy(this.camera.quaternion).invert();
    const axes = [
      ["X", "#ac4240", 1, 0, 0],
      ["Y", "#4265b4", 0, 1, 0],
      ["Z", "#277b50", 0, 0, 1],
    ];
    for (const [label, color, x, y, z] of axes) {
      const v = this.scratch.set(x, y, z).applyQuaternion(inverse);
      const endX = center + v.x * 39, endY = center - v.y * 39;
      ctx.fillStyle = ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(center, center);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(endX, endY, 3.6, 0, Math.PI * 2);
      ctx.fill();
      if (Math.hypot(v.x, v.y) < 0.15) ctx.fillText(label, center + 12, center + 15);
      else ctx.fillText(label, center + v.x * 54, center - v.y * 54);
    }
  }
}
