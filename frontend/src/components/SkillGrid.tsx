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
  onTopTenChange?: (items: string[]) => void;
}

export default function SkillsGrid({ skills, onHighPrioritySkillsChange, onTopTenChange }: SkillsGridProps) {
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

  const calculateTargetIndex = (dragY: number, dragX: number, currentIndex: number) => {
    if (!listRef.current) return currentIndex;
    
    const containerWidth = listRef.current.offsetWidth - 32;
    const itemsPerRow = Math.floor((containerWidth + gutter) / (colSize + gutter));
    
    // Calculate target row and approximate column
    const targetRow = Math.min(Math.max(Math.floor(dragY / (rowSize + gutter)), 0), MAX_ROWS - 1);
    const targetCol = Math.min(Math.floor(dragX / (colSize + gutter)), itemsPerRow - 1);
    
    // Calculate the target index based on row and column
    return (targetRow * itemsPerRow) + targetCol;
  };

  const onDrag = function(this: any) {
    const tile = (this.target as any).tile;
    if (!listRef.current) return;
    
    // Get relative position within the container
    const listBounds = listRef.current.getBoundingClientRect();
    const dragX = this.x;
    const dragY = this.y;
    
    // Get all tiles
    const allTiles = Array.from(listRef.current.children) as HTMLElement[];
    const currentIndex = allTiles.indexOf(this.target);
    
    // Calculate the target index based on drag position
    const targetIndex = calculateTargetIndex(dragY, dragX, currentIndex);
    
    if (currentIndex !== targetIndex) {
      // Remove the dragged element
      this.target.remove();
      
      // Insert at new position
      const referenceElement = allTiles[targetIndex];
      if (referenceElement) {
        listRef.current.insertBefore(this.target, referenceElement);
      } else {
        listRef.current.appendChild(this.target);
      }
      
      // Update layout for all non-dragging tiles
      layoutInvalidated();
    }
    
    tile.inBounds = this.hitTest(listRef.current, 0);
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
        onDragStart: function(this: any) {
          const tile = (this.target as any).tile;
          tile.isDragging = true;
          gsap.to(this.target, {
            zIndex: ++zIndex,
            scale: 1.1,
            boxShadow: shadow2,
            duration: 0.2
          });
        },
        onDrag,
        onRelease: function(this: any) {
          const tile = (this.target as any).tile;
          tile.isDragging = false;

          if (!listRef.current) return;
          
          layoutInvalidated();
          
          gsap.to(this.target, {
            duration: 0.2,
            scale: 1,
            boxShadow: shadow1,
            onComplete: () => {
              tile.positioned = true;
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
    
    // Calculate items per row based on container width
    const containerWidth = listRef.current.offsetWidth - 32;
    const itemsPerRow = Math.floor((containerWidth + gutter) / (colSize + gutter));
    
    tileElements.forEach((element, index) => {
      const tile = (element as any).tile;
      if (tile.isDragging) return;

      const row = Math.floor(index / itemsPerRow);
      const col = index % itemsPerRow;
      
      const xPos = col * (colSize + gutter);
      const yPos = Math.min(row, MAX_ROWS - 1) * (rowSize + gutter);
      
      timeline.to(element, {
        duration: 0.3,
        x: xPos,
        y: yPos,
        ease: "power2.out",
        immediateRender: false
      }, "reflow");
      
      Object.assign(tile, {
        col: col,
        row: row,
        x: xPos,
        y: yPos,
        positioned: true
      });
    });

    // Update high priority skills and top ten
    const topRowItems = tileElements
      .filter((_, index) => Math.floor(index / itemsPerRow) === 0)
      .map(element => element.innerHTML)
      .slice(0, 11);

    onTopTenChange?.(topRowItems);
    
    const newHighPrioritySkills = tileElements
      .filter((_, index) => Math.floor(index / itemsPerRow) === 0)
      .map(element => element.innerHTML);

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