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
  const rowSize = 150;
  const colSize = 150;
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
      colCount = oneColumn ? 1 : Math.floor(listRef.current.offsetWidth / (colSize + gutter));
      gutterStep = colCount == 1 ? gutter : (gutter * (colCount - 1) / colCount);
      rowCount = 0;
      layoutInvalidated();
    };

    const createTile = (skill: string) => {
      const colspan = 1;
      const element = document.createElement('div');
      element.className = 'tile bg-green-500 absolute p-4 font-bold text-gray-800 flex items-center justify-center text-center';
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
          const currentIndex = Array.from(listRef.current?.children || []).indexOf(this.target);
          const dragBounds = this.target.getBoundingClientRect();
          
          // Find tiles that are close to the dragged tile
          const closeTiles = Array.from(listRef.current?.children || [])
            .filter(child => {
              if (child === this.target) return false;
              const box = child.getBoundingClientRect();
              // Only consider tiles within one row distance
              const verticalDistance = Math.abs(box.top - dragBounds.top);
              return verticalDistance <= rowSize * 1.5; // Allow some buffer for dragging between rows
            })
            .map(child => {
              const box = child.getBoundingClientRect();
              const distance = Math.hypot(
                box.left + box.width/2 - (dragBounds.left + dragBounds.width/2),
                box.top + box.height/2 - (dragBounds.top + dragBounds.height/2)
              );
              return { element: child, distance };
            })
            .sort((a, b) => a.distance - b.distance);
          
          if (closeTiles.length > 0) {
            const closest = closeTiles[0];
            const newIndex = Array.from(listRef.current?.children || []).indexOf(closest.element);
            if (currentIndex !== newIndex) {
              changePosition(currentIndex, newIndex);
            }
          }
          
          tile.inBounds = this.hitTest(listRef.current, 0);
          Object.assign(tile, {
            x: this.x,
            y: this.y
          });
        },
        onPress: function(this: any) {
          const tile = (this.target as any).tile;
          tile.isDragging = true;
          tile.lastIndex = tile.index;

          gsap.to(this.target, {
            duration: 0.2,
            opacity: 0.75,
            boxShadow: shadow2,
            scale: 0.95,
            zIndex: '+=1000'
          });
        },
        onRelease: function(this: any) {
          const tile = (this.target as any).tile;
          tile.isDragging = false;

          // Calculate the proper grid position
          const col = Math.round(this.x / (colSize + gutter));
          const row = Math.round(this.y / (rowSize + gutter));
          
          // Force snap to grid
          const xPos = col * (colSize + gutter);
          const yPos = row * (rowSize + gutter);

          // Animate to the nearest grid position
          gsap.to(this.target, {
            duration: 0.2,
            opacity: 1,
            boxShadow: shadow1,
            scale: 1,
            x: xPos,
            y: yPos,
            zIndex: ++zIndex,
            onComplete: () => {
              // Ensure final layout is correct
              layoutInvalidated();
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
    
    // Calculate available width and number of columns
    const containerWidth = listRef.current.offsetWidth;
    colCount = Math.floor((containerWidth + gutter) / (colSize + gutter));
    
    const tileElements = Array.from(listRef.current.getElementsByClassName('tile'));
    
    let col = 0;
    let row = 0;

    tileElements.forEach((element, index) => {
      const tile = (element as any).tile;
      
      if (tile.isDragging) return; // Skip dragging tiles
      
      // Calculate grid position
      if (col >= colCount) {
        col = 0;
        row++;
      }

      // Force exact grid coordinates
      const xPos = col * (colSize + gutter);
      const yPos = row * (rowSize + gutter);

      // Update tile properties
      Object.assign(tile, {
        col: col,
        row: row,
        index: index,
        x: xPos,
        y: yPos,
        width: colSize,
        height: rowSize,
        positioned: true
      });

      // Animate to exact position
      timeline.to(element, 0.3, {
        x: xPos,
        y: yPos,
        ease: "power2.out",
        immediateRender: true
      }, "reflow");

      col++;
    });

    // Update container height
    const totalRows = Math.ceil(tileElements.length / colCount);
    const newHeight = totalRows * rowSize + (totalRows - 1) * gutter;
    timeline.to(listRef.current, 0.2, { height: newHeight }, "reflow");
  };

  return (
    <div 
      ref={listRef}
      className="relative bg-gray-800/20 w-full min-h-[400px] p-4 rounded-lg"
      style={{ 
        display: 'grid',
        gap: `${gutter}px`,
        gridTemplateColumns: `repeat(auto-fill, ${colSize}px)`,
        gridAutoRows: `${rowSize}px`
      }}
    />
  );
}