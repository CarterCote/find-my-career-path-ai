'use client';
import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import EmailCapture from "@/components/EmailCapture";
import Header from "@/components/Header";
import SkillsGrid from "@/components/SkillGrid";

export default function Home() {
  const [currentStep, setCurrentStep] = useState(0);
  const [skills, setSkills] = useState([
    "build", "design/create", "customer/client focus", "persuade/sell", 
    "programming", "solve complex problems", "think critically", 
    "technical expertise", "strategic planning", "adaptability",
    "make decisions", "work in teams", "lead", "manage and develop others",
    "build relationships", "global mindset", "storytelling", "empathy",
    "interpret data", "quantitative", "communicate verbally", 
    "project management", "develop curriculum", "analyze", "research",
    "write", "teach", "social media", "make presentations", 
    "organize", "manage finances"
  ]);



  const handleNext = () => {
    setCurrentStep(prev => prev + 1);
  };

  const handleBack = () => {
    setCurrentStep(prev => prev - 1);
  };

  const fadeVariants = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 }
  };

  return (
    <div className="min-h-screen p-8 pb-20 gap-16 sm:p-2">
      <Header />
      <main className="flex flex-col gap-8 items-center sm:items-start w-full relative min-h-screen">
        <AnimatePresence mode="wait">
          {currentStep === 0 && (
            <motion.div
              key="step0"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">How this works</p>
              <div className="flex flex-col space-y-4">
                <p>Using AI, FindYourCareerPath transforms a user's skills, values, and interests into tailored career pathway recommendations, drawing on real-world examples from individuals with similar backgrounds who are actively pursuing those paths.</p>
                <p>The system then fine-tunes these recommendations to reflect the user's unique preferences, prioritizing the most relevant options.</p>
                <p>As users engage with the suggestions and provide feedback, FindYourCareerPath adapts by learning from this input, continuously refining its recommendations to support the user's career journey as it evolves.</p>
              </div>
              <Button variant="secondary" onClick={handleNext}>Start</Button>
            </motion.div>
          )}

          {currentStep === 1 && (
            <motion.div
              key="step1"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">What makes you thrive?</p>
              <div className="flex flex-col space-y-4">
                <p>To have a meaningful life, it is important to know what matters to you.</p>
                <p>This assessment helps you identify and prioritize what is essential to you at your core, work cultures that fit for you, and skills you enjoy using. This is a practical tool that will give you concrete information you can use to understand and tell your story.</p>
                <p>You will be sorting three categories of cards: Skills, Work Culture, and Core Values.</p>
                <p>Consider what is most important to you in an ideal work situation. Use this frame of reference when you are sorting the cards. You do not need to think about a specific job.</p>
                <p>You get to define what each word or phrase means to you. There is no need to follow the standard definition of a word.</p>
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 2 && (
            <motion.div
              key="step2"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">The Basic Steps</p>
              <div className="flex flex-col space-y-4">
                <p>1. You will be sorting each card, one category at a time, into High, Medium, and Low areas to indicate the level of importance and/or enjoyment for you. The number of cards you put in each area is up to you.</p>
                <p>2. To move the cards around for each category, you can drag them or you can hover over the card and use the arrow menu.</p>
                <p>3. For each category, you'll take your High cards and narrow them down to your top 5.</p>
                <p>4. Next you'll rank order your top 5 from 1 to 5. If you are having trouble ranking the cards, focus on what energizes you. You will come back to these top 5 lists.</p>
                <p>5. At the end of the assessment you will have the opportunity to download a PDF of your results.</p>
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 3 && (
            <motion.div
              className="mx-auto flex-col justify-center space-y-6 items-start py-24 w-[80%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight mb-8">Skills</p>
              <SkillsGrid skills={skills} />
              <div className="flex flex-row items-center w-full justify-between mt-8">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 4 && (
            <motion.div
              key="step4"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">Core Values</p>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
