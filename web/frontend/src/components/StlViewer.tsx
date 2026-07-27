import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { RotateCcw } from "lucide-react";

export function StlViewer({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null);
  const reset = useRef<(() => void) | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!host.current) return;
    const node = host.current;
    setError("");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#132b2a");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    node.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xf5f0df, 0x163332, 2.5));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(2, 3, 4);
    scene.add(light);
    let radius = 1;
    let frame = 0;

    const resize = () => {
      const width = node.clientWidth;
      const height = Math.max(node.clientHeight, 360);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resetCamera = () => {
      const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
      camera.position.set(distance * 0.72, distance * 0.52, distance);
      camera.near = Math.max(radius / 1000, 0.001);
      camera.far = Math.max(radius * 100, 100);
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();
    };
    reset.current = resetCamera;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    resize();
    animate();
    window.addEventListener("resize", resize);

    const loader = new STLLoader();
    loader.load(
      url,
      (geometry) => {
        geometry.center();
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        radius = Math.max(geometry.boundingSphere?.radius ?? 1, 0.001);
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: "#c8e26b",
            roughness: 0.58,
            metalness: 0.08,
          }),
        );
        scene.add(mesh);
        resetCamera();
      },
      undefined,
      () => setError("This STL could not be displayed."),
    );

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          object.material.dispose();
        }
      });
      node.replaceChildren();
    };
  }, [url]);

  return (
    <div className="viewer-shell">
      <div className="stl-viewer" ref={host} />
      <div className="viewer-hint">Drag to rotate · scroll to zoom · right-drag to pan</div>
      <button className="icon-button viewer-reset" onClick={() => reset.current?.()}>
        <RotateCcw size={17} /> Reset view
      </button>
      {error && <div className="viewer-error">{error}</div>}
    </div>
  );
}
