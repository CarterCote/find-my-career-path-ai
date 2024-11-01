import Image from "next/image";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">

        <p className="font-bold">See what's out there.</p>
        <p>Then, gain insight into the path that it takes to get there.</p>
        <p>Pathways.me contains the stories & career possibilities of countless professionals, all of which share the insight that you need to prepare for a 4-year university and beyond.</p>
        <p>(this is not a complete landing page i need to add more here btw)</p>
        <Button className="secondary">See more</Button>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">



      </footer>
    </div>
  );
}
