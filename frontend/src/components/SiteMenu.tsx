import React from 'react';
import Link from 'next/link';
import { useMediaQuery } from 'react-responsive';

const navItems = [

  {
    name: "Take the survey",
    link: "/try",
  },
];

const SiteMenu = ({ useBold = false, vertical = false, textAlign = 'text-center' }) => {
  const isSmallScreen = useMediaQuery({ query: '(max-width: 768px)' });
  const layoutClass = vertical || isSmallScreen ? 'flex-col space-y-8' : 'flex-row space-x-8 items-center';

  return (
    <div className={`${textAlign} flex ${layoutClass} md:text-left`}>
      {navItems.map((item) => (
        item.name === "Take the survey" ? (
          <div className="flex flex-row space-x-2 items-center hover:opacity-60 transition-all duration-500 hover:cursor-pointer">
            <a key={item.link} href={item.link}>
              <h2 className={` text-[15px] transition duration-500`}>{item.name}</h2>
            </a>
            <div className="flex flex-row space-x-2 bg-[#f28d3533] px-2 py-0.5 rounded-full">
              <p className="font-semibold text-xs text-[#f28d35]">Alpha</p>
            </div>
          </div>
        ) : (
          <Link key={item.link} href={item.link}>
            <h2 className={`text-[15px] transition duration-500 hover:text-[#414141]`}>{item.name}</h2>
          </Link>
        )
      ))}
    </div>
  );
};

export default SiteMenu;