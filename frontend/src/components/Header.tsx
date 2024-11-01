"use client";

import { useState, useEffect } from "react";
import { RxHamburgerMenu } from "react-icons/rx";
import { IoMdClose } from "react-icons/io";
import Link from "next/link";
import Image from "next/image";
import SiteMenu from "@/components/SiteMenu";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";


const Header = () => {
    const [isMobile, setIsMobile] = useState(false);
    const [menuOpen, setMenuOpen] = useState(false);

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 768);
        };


        handleResize();
        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <>
            <header className={`flex w-full justify-between bg-primaryBlue items-center flex-row border-b pb-3.5 py-5 md:px-12 px-4 sticky top-0 z-50 'bg-black text-white border-[#242424]`}>
                <div>
                    <Link href="/">
                        <Image
                            src={"/pathway.png"}
                            alt="SX Full Logo"
                            width={136}
                            height={27}
                            className="w-[55%] md:w-[80%] h-auto"
                        />
                    </Link>
                </div>
                {isMobile ? (
                    <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
                        <SheetTrigger asChild>
                            <div className="cursor-pointer">
                                {menuOpen ? <IoMdClose size={30} /> : <RxHamburgerMenu size={24} />}
                            </div>
                        </SheetTrigger>
                        <SheetContent side="top" className="bg-[#111111] w-full p-12 space-y-4">
                            <SiteMenu useBold={true} vertical={true} textAlign="text-left" />
                            {/* <Button href="https://airtable.com/apppnBcY3p3kbfT9V/pagCGeASraULRMoSw/form" variant="dark" className="rounded-xl py-2 px-5 mt-4">
                                <div className={`${messina_semibold.className} text-[12px] md:text-normal tracking-tight`}>
                                    JOIN THE COMMUNITY
                                </div>
                            </Button>
                            <div className="flex flex-col space-y-4 pt-4">
                                <Button href="/login " className="bg-transparent border border-zinc-700 hover:bg-zinc-900 rounded-full items-center justify-center flex flex-row opacity-100 hover:opacity-70 transition duration-200 hover:cursor-pointer">
                                    <p className={`tracking-tighter text-sm text-white`}>
                                        LOG IN
                                    </p>
                                </Button>
                                <div className="relative inline-block w-full">
                                    <Button 
                                        href="https://airtable.com/apppnBcY3p3kbfT9V/pagCGeASraULRMoSw/form"
                                    className="opacity-70 w-full px-14 hover:opacity-100 rounded-full bg-transparent border border-transparent relative transition-all duration-200 h-10"
                                    style={{
                                    background: 'linear-gradient(#17171a,#17171a) padding-box, linear-gradient(180deg,hsla(0,0%,100%,.5),hsla(0,0%,100%,.25)) border-box'
                                        }}

                                    />
                                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                        <p className={` tracking-tighter text-sm text-white`}>
                                            SIGN UP
                                        </p>
                                    </div>
                                </div>
                            </div> */}
                        </SheetContent>
                    </Sheet>
                ) : (
                    <>
                        <SiteMenu useBold={false} />
                        {/* <div className="flex flex-row space-x-6">
                            <a href="/login " className="items-center justify-center flex flex-row opacity-100 hover:opacity-70 transition duration-200 hover:cursor-pointer">
                                <p className={` tracking-tighter text-sm text-white`}>
                                    LOG IN
                                </p>
                            </a>
                            <div className="relative inline-block">
                                <Button 
                                    href="https://airtable.com/apppnBcY3p3kbfT9V/pagCGeASraULRMoSw/form"
                                    className="opacity-70 px-14 hover:opacity-100 rounded-full bg-transparent border border-transparent relative transition-all duration-200 w-24 h-10"
                                    style={{
                                    background: 'linear-gradient(#17171a,#17171a) padding-box, linear-gradient(180deg,hsla(0,0%,100%,.5),hsla(0,0%,100%,.25)) border-box'
                                    }}
                                >
                                </Button>
                                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                    <p className={`tracking-tighter text-sm text-white`}>
                                        SIGN UP
                                    </p>
                                </div>
                            </div>
                        </div> */}
                    </>
                )}
            </header>
        </>
    );
};

export default Header;