import Image from "next/image";
import { Button } from "@/components/ui/button";
import EmailCapture from "@/components/EmailCapture";
import Header from "@/components/Header";

export default function Home() {
  return (
    <div className="min-h-screen p-8 pb-20 gap-16 sm:p-2">
      <Header />
      <main className="flex flex-col gap-8 items-center sm:items-start max-w-6xl mx-auto py-24 relative min-h-screen">
        <div className="min-h-screen">
          <div className="absolute top-0 right-0 bottom-0 w-1/3">
            <Image 
              src="/heroCardGrid.png" 
              alt="Career pathways illustration" 
              fill
              className="object-contain object-right"
              priority
            />
          </div>
          <div className="flex flex-col space-y-6 w-2/3">
            <Image src="/logo.png" alt="" width={400} height={24} className="pb-12" />
            <p className="font-circular font-bold text-5xl tracking-tight">See what's <span className="font-planet tracking-wide text-fadedGold">possible.</span></p>
            <div className="flex flex-col space-y-1">
              <p className="">Then, gain insight into the path that it takes to get there.</p>
              <p>Pathways.me contains the stories & career possibilities of countless professionals, all of which share the insight that you need to prepare for a 4-year university and beyond.</p>
            </div>
            <EmailCapture />
          </div>
        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">



      </footer>
    </div>
  );
}
