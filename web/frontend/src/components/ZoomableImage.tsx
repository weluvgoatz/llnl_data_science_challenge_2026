import { PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";

const MIN_SCALE = 1;
const MAX_SCALE = 6;

export function ZoomableImage({ src, alt }: { src: string; alt: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const drag = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null);

  const zoomBy = (factor: number) => {
    setScale((current) => {
      const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, current * factor));
      if (next === MIN_SCALE) setOffset({ x: 0, y: 0 });
      return next;
    });
  };

  const reset = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  // Reset zoom whenever the underlying image changes (e.g. the slice slider moves).
  useEffect(reset, [src]);

  // React attaches wheel listeners passively by default, which silently ignores
  // preventDefault; a native listener is required to stop the page from
  // scrolling while zooming the image.
  useEffect(() => {
    const node = hostRef.current;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomBy(Math.exp(-event.deltaY * 0.0018));
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (scale <= MIN_SCALE) return;
    drag.current = { startX: event.clientX, startY: event.clientY, originX: offset.x, originY: offset.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    setOffset({
      x: drag.current.originX + (event.clientX - drag.current.startX),
      y: drag.current.originY + (event.clientY - drag.current.startY),
    });
  };
  const endDrag = () => {
    drag.current = null;
  };

  return (
    <div
      className={`zoomable-image ${scale > MIN_SCALE ? "zoomed" : ""}`}
      ref={hostRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
      onDoubleClick={reset}
    >
      <img
        src={src}
        alt={alt}
        draggable={false}
        style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }}
      />
      <div className="zoom-controls">
        <button aria-label="Zoom out" onClick={() => zoomBy(1 / 1.4)}><Minus size={14} /></button>
        <button aria-label="Zoom in" onClick={() => zoomBy(1.4)}><Plus size={14} /></button>
        {scale > MIN_SCALE && (
          <button aria-label="Reset zoom" onClick={reset}><RotateCcw size={14} /></button>
        )}
      </div>
    </div>
  );
}
