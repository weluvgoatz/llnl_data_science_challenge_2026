import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RotateCcw } from "lucide-react";

interface DesignJunction {
  id: number;
  position: [number, number, number];
}

interface DesignStrut {
  id: number;
  junction0: number;
  junction1: number;
}

interface DesignJson {
  junctions: DesignJunction[];
  struts: DesignStrut[];
}

interface ViewerState {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  renderer: THREE.WebGLRenderer;
  lines: THREE.LineSegments | null;
  radius: number;
  resetCamera: () => void;
}

// Renders the as-designed lattice geometry straight from the uploaded design
// JSON's own junctions/struts -- the intended shape, before (or without)
// any analysis. Every coordinate is the design file's own data; nothing
// here is measured, classified, or invented. Modeled on DefectViewer's
// scene setup, simplified: no verdicts to color by, no picking.
export function DesignGraphViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<ViewerState | null>(null);
  const [error, setError] = useState("");
  const [strutCount, setStrutCount] = useState<number | null>(null);

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

    const ctx: ViewerState = { scene, camera, controls, renderer, lines: null, radius: 1, resetCamera: () => {} };
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
    setError("");
    setStrutCount(null);
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`Request failed (${response.status})`);
        return response.json();
      })
      .then((data: DesignJson) => {
        const ctx = state.current;
        if (!ctx || !data.junctions || !data.struts) throw new Error("Not a recognizable design graph");

        const positionById = new Map(data.junctions.map((j) => [j.id, j.position]));
        const positions: number[] = [];
        const min = new THREE.Vector3(Infinity, Infinity, Infinity);
        const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

        for (const strut of data.struts) {
          const a = positionById.get(strut.junction0);
          const b = positionById.get(strut.junction1);
          if (!a || !b) continue;
          positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
          min.set(Math.min(min.x, a[0], b[0]), Math.min(min.y, a[1], b[1]), Math.min(min.z, a[2], b[2]));
          max.set(Math.max(max.x, a[0], b[0]), Math.max(max.y, a[1], b[1]), Math.max(max.z, a[2], b[2]));
        }
        if (positions.length === 0) throw new Error("Design graph has no renderable struts");

        const center = min.clone().add(max).multiplyScalar(0.5);
        for (let i = 0; i < positions.length; i += 3) {
          positions[i] -= center.x;
          positions[i + 1] -= center.y;
          positions[i + 2] -= center.z;
        }

        if (ctx.lines) {
          ctx.scene.remove(ctx.lines);
          ctx.lines.geometry.dispose();
          (ctx.lines.material as THREE.Material).dispose();
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        const material = new THREE.LineBasicMaterial({ color: "#c8e26b" });
        const lines = new THREE.LineSegments(geometry, material);
        ctx.scene.add(lines);
        ctx.lines = lines;
        ctx.radius = Math.max(max.clone().sub(min).length() / 2, 0.001);
        ctx.resetCamera();

        setStrutCount(data.struts.length);
      })
      .catch(() => setError("This design JSON could not be rendered as a lattice graph."));
  }, [url]);

  return (
    <div className="viewer-shell">
      <div className="stl-viewer" ref={host} />
      <div className="viewer-hint">
        Drag to rotate · scroll to zoom · right-drag to pan
        {strutCount !== null ? ` · ${strutCount.toLocaleString()} designed struts (as-designed, not yet inspected)` : ""}
      </div>
      <button className="icon-button viewer-reset" onClick={() => state.current?.resetCamera()}>
        <RotateCcw size={17} /> Reset view
      </button>
      {error && <div className="viewer-error">{error}</div>}
    </div>
  );
}
