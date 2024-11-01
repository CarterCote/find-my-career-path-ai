import React from 'react';
import Link from 'next/link';
import { useMediaQuery } from 'react-responsive';

const navItems = [
  {
    name: "BUILDER PODS",
    link: "/community",
  },
  {
    name: "CAMPUSES",
    link: "/campuses",
  },
  {
    name: "EVENTS",
    link: "/events",
  },
  {
    name: "DONATE",
    link: "/",
  },
  {
    name: "RESOURCES",
    link: "/resources",
  },
  {
    name: "ABOUT",
    link: "/about",
  },
];

const SiteMenu = ({ useBold = false, vertical = false, textAlign = 'text-center' }) => {
  const isSmallScreen = useMediaQuery({ query: '(max-width: 768px)' });
  const layoutClass = vertical || isSmallScreen ? 'flex-col space-y-8' : 'flex-row space-x-8 items-center';

  return (
    <div className={`${textAlign} flex ${layoutClass} md:text-left`}>
      {navItems.map((item) => (
        item.name === "DONATE" ? (
          <a key={item.link} href={item.link} target="_blank" rel="noopener noreferrer">
            <h2 className={` text-[15px] transition duration-500 hover:text-[#414141]`}>{item.name}</h2>
          </a>
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