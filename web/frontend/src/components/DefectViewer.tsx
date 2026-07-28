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
  lines: THREE.LineSegments | null;
  radius: number;
  resetCamera: () => void;
}

export function DefectViewer({
  struts,
  visibleVerdicts,
}: {
  struts: Strut[];
  visibleVerdicts: Set<StrutVerdict>;
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
      lines: null,
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

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      controls.dispose();
      renderer.dispose();
      if (ctx.lines) {
        ctx.lines.geometry.dispose();
        (ctx.lines.material as THREE.Material).dispose();
      }
      node.replaceChildren();
      state.current = null;
    };
  }, []);

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
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    const color = new THREE.Color();

    for (const strut of struts) {
      if (!visibleVerdicts.has(strut.verdict)) continue;
      const [x0, y0, z0] = strut.p0;
      const [x1, y1, z1] = strut.p1;
      positions.push(x0, y0, z0, x1, y1, z1);
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

  return (
    <div className="viewer-shell defect-viewer-shell">
      <div className="stl-viewer" ref={host} />
      <div className="viewer-hint">Drag to rotate · scroll to zoom · right-drag to pan</div>
      <button className="icon-button viewer-reset" onClick={() => state.current?.resetCamera()}>
        <RotateCcw size={17} /> Reset view
      </button>
      {empty && <div className="viewer-error">No struts match the current filters.</div>}
    </div>
  );
}
