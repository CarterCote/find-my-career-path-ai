import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { gsap } from 'gsap';
import Draggable from 'gsap/Draggable';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(Draggable);
}

interface Tile {
  col: number | null;
  colspan: number;
  height: number;
  inBounds: boolean;
  index: number | null;
  isDragging: boolean;
  lastIndex: number | null;
  newTile: boolean;
  positioned: boolean;
  row: number | null;
  rowspan: number;
  width: number;
  x: number;
  y: number;
}

interface SkillsGridProps {
  skills: string[];
}

export default function SkillsGrid({ skills }: SkillsGridProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [tiles, setTiles] = useState<HTMLElement[]>([]);
  
  // Add lastX ref to track drag direction
  const lastXRef = useRef(0);

  // Grid options
  const rowSize = 100;
  const colSize = 100;
  const gutter = 10;
  const threshold = "50%";
  const fixedSize = false;
  const oneColumn = false;
  
  let colCount = 0;
  let rowCount = 0;
  let gutterStep = 0;
  let zIndex = 1000;

  const shadow1 = "0 1px 3px 0 rgba(0, 0, 0, 0.5), 0 1px 2px 0 rgba(0, 0, 0, 0.6)";
  const shadow2 = "0 6px 10px 0 rgba(0, 0, 0, 0.3), 0 2px 2px 0 rgba(0, 0, 0, 0.2)";

  const MAX_ROWS = 4;

  const changePosition = (from: number, to: number, rowToUpdate = -1) => {
    if (!listRef.current) return;
    
    const tileElements = Array.from(listRef.current.getElementsByClassName('tile'));
    // Validate indices are within bounds
    if (from < 0 || to < 0 || from >= tileElements.length || to >= tileElements.length) return;
    
    const element = tileElements[from];
    const target = tileElements[to];

    // Ensure both elements exist
    if (!element || !target) return;

    // Get the parent container
    const container = listRef.current;
    if (!container) return;

    // Remove and reinsert the element at the new position
    element.remove();
    if (from > to) {
      container.insertBefore(element, target);
    } else {
      const nextSibling = target.nextSibling;
      container.insertBefore(element, nextSibling);
    }

    layoutInvalidated(rowToUpdate);
  };

  useEffect(() => {
    if (!listRef.current) return;
    
    const resize = () => {
      if (!listRef.current) return;
      const containerWidth = listRef.current.offsetWidth - 32; // Account for padding
      // Calculate how many columns can fit in the container width
      colCount = Math.floor((containerWidth + gutter) / (colSize + gutter));
      layoutInvalidated();
    };

    const createTile = (skill: string) => {
      const colspan = 1;
      const element = document.createElement('div');
      element.className = 'tile bg-tertiaryBlue absolute p-4 font-bold text-gray-800 leading-[100%] text-sm flex items-center justify-center text-center';
      element.style.width = `${colSize}px`;
      element.style.height = `${rowSize}px`;
      element.innerHTML = skill;

      const tile: Tile = {
        col: null,
        colspan: colspan,
        height: rowSize,
        inBounds: true,
        index: null,
        isDragging: false,
        lastIndex: null,
        newTile: true,
        positioned: false,
        row: null,
        rowspan: 1,
        width: colSize,
        x: 0,
        y: 0
      };

      (element as any).tile = tile;

      if (listRef.current) {
        listRef.current.appendChild(element);
        setTiles(prev => [...prev, element]);
      }

      Draggable.create(element, {
        type: 'x,y',
        onDrag: function(this: any) {
          const tile = (this.target as any).tile;
          const dragBounds = this.target.getBoundingClientRect();
          
          const listElement = listRef.current;
          if (!listElement) return;
          
          // Determine which row we're hovering over
          const currentY = dragBounds.top - listElement.getBoundingClientRect().top;
          const hoveringRow = Math.min(Math.max(Math.floor(currentY / (rowSize + gutter)), 0), MAX_ROWS - 1);
          
          // Find tiles in the same row
          const tilesInRow = Array.from(listElement.children)
            .filter(child => {
              if (child === this.target) return false;
              const box = child.getBoundingClientRect();
              const childY = box.top - listElement.getBoundingClientRect().top;
              const childRow = Math.floor(childY / (rowSize + gutter));
              return childRow === hoveringRow;
            });
          
          // Find closest tile in row based on horizontal position
          const closestTile = tilesInRow
            .map(child => {
              const box = child.getBoundingClientRect();
              const distance = Math.abs(box.left - dragBounds.left);
              return { element: child, distance };
            })
            .sort((a, b) => a.distance - b.distance)[0];
          
          if (closestTile) {
            const currentIndex = Array.from(listElement.children).indexOf(this.target);
            const newIndex = Array.from(listElement.children).indexOf(closestTile.element);
            
            // Only swap if we're in the target row
            if (currentIndex !== newIndex) {
              changePosition(currentIndex, newIndex, hoveringRow);
            }
          }
          
          tile.inBounds = this.hitTest(listElement, 0);
          Object.assign(tile, {
            x: this.x,
            y: this.y
          });
        },
        onRelease: function(this: any) {
          const tile = (this.target as any).tile;
          tile.isDragging = false;

          if (!listRef.current) return;
          
          // Determine final row position
          const currentY = this.y;
          const finalRow = Math.min(Math.max(Math.floor(currentY / (rowSize + gutter)), 0), MAX_ROWS - 1);
          
          // Get current index and calculate new position
          const currentIndex = Array.from(listRef.current.children).indexOf(this.target);
          
          // Calculate the nearest column position based on x coordinate
          const nearestCol = Math.min(
            Math.max(Math.round(this.x / (colSize + gutter)), 0),
            colCount - 1
          );
          
          // Calculate exact grid position
          const xPos = nearestCol * (colSize + gutter);
          const yPos = finalRow * (rowSize + gutter);

          gsap.to(this.target, {
            duration: 0.2,
            opacity: 1,
            boxShadow: shadow1,
            scale: 1,
            x: xPos,
            y: yPos,
            zIndex: ++zIndex,
            onComplete: () => {
              layoutInvalidated(finalRow);
            }
          });
        }
      });
    };

    // Initialize
    window.addEventListener('resize', resize);
    skills.forEach(createTile);
    resize();

    return () => {
      window.removeEventListener('resize', resize);
    };
  }, []);

  const layoutInvalidated = (rowToUpdate = -1) => {
    if (!listRef.current) return;
    
    const timeline = gsap.timeline();
    const tileElements = Array.from(listRef.current.getElementsByClassName('tile'));
    
    // Calculate positions based on container width
    tileElements.forEach((element, index) => {
      const tile = (element as any).tile;
      if (tile.isDragging) return;

      // Calculate row and column based on index
      const row = Math.min(Math.floor(index / colCount), MAX_ROWS - 1);
      const col = index % colCount;
      
      // Calculate exact pixel positions
      const xPos = col * (colSize + gutter);
      const yPos = row * (rowSize + gutter);
      
      Object.assign(tile, {
        col: col,
        row: row,
        x: xPos,
        y: yPos,
        positioned: true
      });
      
      timeline.to(element, 0.3, {
        x: xPos,
        y: yPos,
        ease: "power2.out",
        immediateRender: true
      }, "reflow");
    });
  };

  return (
    <div 
      ref={listRef}
      className="relative bg-gray-800/20 w-full rounded-lg p-4"
      style={{ 
        height: `${(MAX_ROWS * rowSize) + ((MAX_ROWS - 1) * gutter) + 32}px`,
        position: 'relative'
      }}
    />
  );
}