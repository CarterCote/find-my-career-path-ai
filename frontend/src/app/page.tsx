import Image from "next/image";
import { Button } from "@/components/ui/button";
import EmailCapture from "@/components/EmailCapture";
import Header from "@/components/Header";

export default function Home() {
  return (
    <div className="min-h-screen p-8 pb-20 gap-16 sm:p-2">
      <Header />
      <main className="flex flex-col gap-8 items-center sm:items-start w-full relative min-h-screen">
        <div className="min-h-[80vh] bg-[url('/heroCardGridWide.png')] bg-cover w-full bg-center bg-no-repeat relative">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/50"></div>
          <div className="flex flex-col min-h-[80vh] space-y-6 w-3/4 mx-auto justify-center relative z-10">
            <Image src="/pathway.png" alt="" width={400} height={24} className="pb-12" />
            <p className="font-circular font-bold text-5xl tracking-tight">Discover what's <span className="font-planet tracking-wide text-fadedGold">possible.</span></p>
            <div className="flex flex-col space-y-1 w-1/2">
              <p className="">Explore personalized career pathways and the steps to achieve them.</p>
              <p>
                Pathways.me brings you the stories and career journeys of countless professionals, offering the insights you need to navigate your path through university and beyond.
              </p>
            </div>
            <EmailCapture />
          </div>
        </div>
        <div className="flex flex-row w-1/2 mx-auto py-36 space-x-16 items-center align-center">
          <div className="w-2/3">
            <h2 className="font-circular font-bold text-4xl mb-4">How it works</h2>
            <Image 
              src="/howItWorks.png" 
              alt="How it works illustration" 
              width={1200} 
              height={1200}
              className="w-full h-auto"
              priority
            />
          </div>
          <div className="flex flex-col space-y-8 w-full">
            <div className="flex flex-col space-y-6">
              <p>Using AI, FindYourCareerPath transforms a user's skills, values, and interests into tailored career pathway recommendations, drawing on real-world examples from individuals with similar backgrounds who are actively pursuing those paths. </p>
              
              <p>The system then fine-tunes these recommendations to reflect the user’s unique preferences, prioritizing the most relevant options.</p>
              
              <p>As users engage with the suggestions and provide feedback, FindYourCareerPath adapts by learning from this input, continuously refining its recommendations to support the user’s career journey as it evolves.</p>
            </div>

            <div className="flex gap-4">
              <Button variant="outline" size="outline">Start exploring →</Button>
            </div>
          </div>
        </div>
        <div className="w-1/3 mx-auto flex-col justify-center space-y-6 items-center text-center py-24">
          <p className="font-circular font-bold text-4xl tracking-tight">Our mission</p>
          <div className="flex flex-col space-y-1">
            <p>Pathway.me aims to democratize access to career path possibilities to high schoolers everywhere. Every student deserves a fair chance to see what's possible, and to know about what it takes to pursue that path at a university and beyond. </p>
          </div>
          <Button variant="outline" size="outline">Read more</Button>
        </div>
        <div className="w-1/3 mx-auto flex-col justify-center space-y-4 items-center align-center text-center py-24">
          <p className="font-circular font-bold text-4xl tracking-tight">Sign up for early access</p>
          <div className="flex flex-col space-y-1">
            <p>Learn about the project's progress, and gain access to the platform by entering your email below.</p>
          </div>
          <div className="flex w-full justify-center">
          <EmailCapture />
          </div>

        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">



      </footer>
    </div>
  );
}
