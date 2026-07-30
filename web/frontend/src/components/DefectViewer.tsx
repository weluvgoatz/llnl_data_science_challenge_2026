import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RotateCcw } from "lucide-react";
import { VERDICT_COLORS } from "../defects";
import type { Strut, StrutVerdict } from "../types";

interface ViewerState {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  renderer: THREE.WebGLRenderer;
  raycaster: THREE.Raycaster;
  lines: THREE.LineSegments | null;
  highlight: THREE.Points | null;
  segmentStrutIds: number[];
  center: THREE.Vector3;
  radius: number;
  resetCamera: () => void;
  onSelectStrut?: (strutId: number) => void;
}

export function DefectViewer({
  struts,
  visibleVerdicts,
  selectedStrutIds,
  onSelectStrut,
}: {
  struts: Strut[];
  visibleVerdicts: Set<StrutVerdict>;
  selectedStrutIds?: Set<number>;
  onSelectStrut?: (strutId: number) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<ViewerState | null>(null);
  const [empty, setEmpty] = useState(false);

  useEffect(() => {
    if (!host.current) return;
    const node = host.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#132b2a");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1e6);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    node.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ctx: ViewerState = {
      scene,
      camera,
      controls,
      renderer,
      raycaster: new THREE.Raycaster(),
      lines: null,
      highlight: null,
      segmentStrutIds: [],
      center: new THREE.Vector3(),
      radius: 1,
      resetCamera: () => {},
    };
    ctx.resetCamera = () => {
      const distance = ctx.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
      camera.position.set(distance * 0.72, distance * 0.52, distance);
      camera.near = Math.max(ctx.radius / 1000, 0.001);
      camera.far = Math.max(ctx.radius * 100, 100);
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();
    };
    state.current = ctx;

    const resize = () => {
      const width = node.clientWidth;
      const height = Math.max(node.clientHeight, 360);
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    let frame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    resize();
    ctx.resetCamera();
    animate();
    window.addEventListener("resize", resize);

    // Manual click detection (not the browser's "click") so an orbit-drag
    // release is never mistaken for a strut pick.
    let downPos: { x: number; y: number } | null = null;
    const onPointerDown = (event: PointerEvent) => {
      downPos = { x: event.clientX, y: event.clientY };
    };
    const onPointerUp = (event: PointerEvent) => {
      const start = downPos;
      downPos = null;
      if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
      if (!ctx.lines || !ctx.onSelectStrut) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      ctx.raycaster.params.Line = { threshold: Math.max(ctx.radius * 0.01, 2) };
      ctx.raycaster.setFromCamera(ndc, camera);
      const hits = ctx.raycaster.intersectObject(ctx.lines);
      if (!hits.length) return;
      const segmentIndex = Math.floor((hits[0].index ?? 0) / 2);
      const strutId = ctx.segmentStrutIds[segmentIndex];
      if (strutId !== undefined) ctx.onSelectStrut(strutId);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointerup", onPointerUp);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      controls.dispose();
      renderer.dispose();
      if (ctx.lines) {
        ctx.lines.geometry.dispose();
        (ctx.lines.material as THREE.Material).dispose();
      }
      if (ctx.highlight) {
        ctx.highlight.geometry.dispose();
        (ctx.highlight.material as THREE.Material).dispose();
      }
      node.replaceChildren();
      state.current = null;
    };
  }, []);

  // Keep the latest callback available to the pointerup handler above
  // without re-registering the listener on every render.
  useEffect(() => {
    if (state.current) state.current.onSelectStrut = onSelectStrut;
  }, [onSelectStrut]);

  useEffect(() => {
    const ctx = state.current;
    if (!ctx) return;
    if (ctx.lines) {
      ctx.scene.remove(ctx.lines);
      ctx.lines.geometry.dispose();
      (ctx.lines.material as THREE.Material).dispose();
      ctx.lines = null;
    }

    const positions: number[] = [];
    const colors: number[] = [];
    const segmentStrutIds: number[] = [];
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    const color = new THREE.Color();

    for (const strut of struts) {
      if (!visibleVerdicts.has(strut.verdict)) continue;
      const [x0, y0, z0] = strut.p0;
      const [x1, y1, z1] = strut.p1;
      positions.push(x0, y0, z0, x1, y1, z1);
      segmentStrutIds.push(strut.id);
      color.set(VERDICT_COLORS[strut.verdict]);
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      min.set(Math.min(min.x, x0, x1), Math.min(min.y, y0, y1), Math.min(min.z, z0, z1));
      max.set(Math.max(max.x, x0, x1), Math.max(max.y, y0, y1), Math.max(max.z, z0, z1));
    }

    setEmpty(positions.length === 0);
    if (positions.length === 0) return;

    const center = min.clone().add(max).multiplyScalar(0.5);
    for (let index = 0; index < positions.length; index += 3) {
      positions[index] -= center.x;
      positions[index + 1] -= center.y;
      positions[index + 2] -= center.z;
    }
    ctx.center = center;
    ctx.segmentStrutIds = segmentStrutIds;
    ctx.radius = Math.max(max.clone().sub(min).length() / 2, 0.001);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({ vertexColors: true });
    const lines = new THREE.LineSegments(geometry, material);
    ctx.scene.add(lines);
    ctx.lines = lines;
    ctx.resetCamera();
  }, [struts, visibleVerdicts]);

  // Selection markers: bright, camera-facing points at each selected strut's
  // midpoint, in the same centered coordinate frame as the base geometry
  // above (recomputed together whenever it is, so they never drift apart).
  useEffect(() => {
    const ctx = state.current;
    if (!ctx) return;
    if (ctx.highlight) {
      ctx.scene.remove(ctx.highlight);
      ctx.highlight.geometry.dispose();
      (ctx.highlight.material as THREE.Material).dispose();
      ctx.highlight = null;
    }
    if (!selectedStrutIds || selectedStrutIds.size === 0) return;

    const positions: number[] = [];
    for (const strut of struts) {
      if (!selectedStrutIds.has(strut.id)) continue;
      const mx = (strut.p0[0] + strut.p1[0]) / 2 - ctx.center.x;
      const my = (strut.p0[1] + strut.p1[1]) / 2 - ctx.center.y;
      const mz = (strut.p0[2] + strut.p1[2]) / 2 - ctx.center.z;
      positions.push(mx, my, mz);
    }
    if (positions.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: "#ffffff",
      size: Math.max(ctx.radius * 0.025, 3),
      sizeAttenuation: true,
      depthTest: false,
    });
    const points = new THREE.Points(geometry, material);
    points.renderOrder = 999;
    ctx.scene.add(points);
    ctx.highlight = points;
  }, [struts, selectedStrutIds, visibleVerdicts]);

  return (
    <div className="viewer-shell defect-viewer-shell">
      <div className="stl-viewer" ref={host} />
      <div className="viewer-hint">
        Drag to rotate · scroll to zoom · right-drag to pan{onSelectStrut ? " · click a strut to select it" : ""}
      </div>
      <button className="icon-button viewer-reset" onClick={() => state.current?.resetCamera()}>
        <RotateCcw size={17} /> Reset view
      </button>
      {empty && <div className="viewer-error">No struts match the current filters.</div>}
    </div>
  );
}
