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
  onHighPrioritySkillsChange?: (skills: string[]) => void;
}

export default function SkillsGrid({ skills, onHighPrioritySkillsChange }: SkillsGridProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [tiles, setTiles] = useState<HTMLElement[]>([]);
  
  // Add lastX ref to track drag direction
  const lastXRef = useRef(0);

  // Grid options
  const rowSize = 100;
  const colSize = 100;
  const gutter = 12;
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
    
    // Clear existing tiles
    while (listRef.current.firstChild) {
      listRef.current.removeChild(listRef.current.firstChild);
    }
    setTiles([]);

    const resize = () => {
      if (!listRef.current) return;
      const containerWidth = listRef.current.offsetWidth - 32;
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
          
          if (!listRef.current) return;
          
          // Determine which row we're hovering over
          const currentY = dragBounds.top - listRef.current.getBoundingClientRect().top;
          const hoveringRow = Math.min(Math.max(Math.floor(currentY / (rowSize + gutter)), 0), MAX_ROWS - 1);
          
          // Only look at tiles in the current hovering row
          const tilesInRow = getItemsInRow(hoveringRow);
          
          // Find closest tile in row based on horizontal position
          const closestTile = tilesInRow
            .filter(child => child !== this.target)
            .map(child => {
              const box = child.getBoundingClientRect();
              const distance = Math.abs(box.left - dragBounds.left);
              return { element: child, distance };
            })
            .sort((a, b) => a.distance - b.distance)[0];
          
          if (closestTile) {
            const currentIndex = tilesInRow.indexOf(this.target);
            const newIndex = tilesInRow.indexOf(closestTile.element);
            
            // Only reorder within the same row
            if (currentIndex !== -1 && newIndex !== -1) {
              reorderWithinRow(hoveringRow, currentIndex, newIndex);
            }
          }
          
          tile.inBounds = this.hitTest(listRef.current, 0);
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
          
          // Get items in the final row
          const tilesInRow = getItemsInRow(finalRow);
          
          // Calculate nearest position in the row
          const nearestCol = Math.min(
            Math.max(Math.round(this.x / (colSize + gutter)), 0),
            tilesInRow.length
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
              tile.row = finalRow;
              layoutInvalidated(finalRow);
            }
          });
        }
      });
    };

    // Create new tiles
    skills.forEach(createTile);
    resize();

    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      // Clean up tiles on unmount
      if (listRef.current) {
        while (listRef.current.firstChild) {
          listRef.current.removeChild(listRef.current.firstChild);
        }
      }
    };
  }, [skills]);

  const [highPrioritySkills, setHighPrioritySkills] = useState<string[]>([]);

  const layoutInvalidated = (rowToUpdate = -1) => {
    if (!listRef.current) return;
    
    const timeline = gsap.timeline();
    const tileElements = Array.from(listRef.current.getElementsByClassName('tile'));
    
    // Track items that end up in row 0 (high priority)
    const newHighPrioritySkills: string[] = [];
    
    tileElements.forEach((element, index) => {
      const tile = (element as any).tile;
      if (tile.isDragging) return;

      const row = Math.min(Math.floor(index / colCount), MAX_ROWS - 1);
      const col = index % colCount;
      
      const xPos = col * (colSize + gutter);
      const yPos = row * (rowSize + gutter);
      
      Object.assign(tile, {
        col: col,
        row: row,
        x: xPos,
        y: yPos,
        positioned: true
      });
      
      // If this item is in row 0, add it to high priority skills
      if (row === 0) {
        newHighPrioritySkills.push(element.innerHTML);
      }
      
      timeline.to(element, 0.3, {
        x: xPos,
        y: yPos,
        ease: "power2.out",
        immediateRender: true
      }, "reflow");
    });

    // Update high priority skills and notify parent
    if (JSON.stringify(newHighPrioritySkills) !== JSON.stringify(highPrioritySkills)) {
      setHighPrioritySkills(newHighPrioritySkills);
      onHighPrioritySkillsChange?.(newHighPrioritySkills);
    }
  };

  // Add a function to get items in a specific row
  const getItemsInRow = (row: number) => {
    if (!listRef.current) return [];
    return Array.from(listRef.current.children).filter(child => {
      const tile = (child as any).tile;
      return tile.row === row;
    });
  };

  // Add a function to reorder items within a row
  const reorderWithinRow = (row: number, fromIndex: number, toIndex: number) => {
    if (!listRef.current) return;
    
    const tilesInRow = getItemsInRow(row);
    if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0 || 
        fromIndex >= tilesInRow.length || toIndex >= tilesInRow.length) return;
    
    const element = tilesInRow[fromIndex];
    const target = tilesInRow[toIndex];
    
    if (!element || !target || !target.parentNode) return;
    
    // Reorder only within the row
    if (fromIndex > toIndex) {
      target.parentNode.insertBefore(element, target);
    } else {
      target.parentNode.insertBefore(element, target.nextSibling);
    }
    
    layoutInvalidated(row);
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