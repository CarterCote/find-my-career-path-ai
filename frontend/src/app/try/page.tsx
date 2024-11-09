'use client';
import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import EmailCapture from "@/components/EmailCapture";
import Header from "@/components/Header";
import SkillsGrid from "@/components/SkillGrid";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { Textarea } from "@/components/ui/textarea";


export default function Try() {
    const [currentStep, setCurrentStep] = useState(0);
    const [dialogOpen, setDialogOpen] = useState(false);
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
    ].filter((value, index, self) => self.indexOf(value) === index));

    const [workCulture, setWorkCulture] = useState([
        "inspiring", "fast-paced", "growth potential", "ethical leadership",
        "collaboration", "entrepreneurial", "innovation", "professional development",
        "engaging work", "geographical location", "flexibility", "service focus",
        "respect", "workplace surroundings", "fairness", "supportive environment",
        "transparency", "mentoring", "variety", "travel", "challenging",
        "predictability", "humor", "inclusive workplace", "salary", "fun",
        "stability", "recognition", "structure", "life balance", "competition"
    ].filter((value, index, self) => self.indexOf(value) === index));

    const [coreValues, setCoreValues] = useState([
        "integrity", "personal development", "family/friends", "fulfilling one's potential",
        "authenticity", "impact", "creativity", "excellence", "intellectual challenge",
        "success", "achievement", "community", "gratitude", "learning", "social justice",
        "autonomy", "compassion", "relationships", "safety", "diversity", "joy",
        "belonging", "environmental responsibility", "adventure", "helping", "prestige",
        "harmony", "spirituality", "trust", "well-being", "simplicity"
    ].filter((value, index, self) => self.indexOf(value) === index));

    const [highSkills, setHighSkills] = useState<string[]>([]);
    const [topFiveSkills, setTopFiveSkills] = useState<string[]>([]);
    const [step3HighSkills, setStep3HighSkills] = useState<string[]>([]);
    const getHighPrioritySkills = () => {
        return step3HighSkills;
    };

    const handleNext = () => {
        if (currentStep === 2) {
        setDialogOpen(true);
        }
        if (currentStep === 3) {
        const highPrioritySkills = getHighPrioritySkills();
        setHighSkills(highPrioritySkills);
        }
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
                <p>Using AI, Pathways transforms a user's skills, values, and interests into tailored career pathway recommendations, drawing on real-world examples from individuals with similar backgrounds who are actively pursuing those paths.</p>
                <p>The system then fine-tunes these recommendations to reflect the user's unique preferences, prioritizing the most relevant options.</p>
                <p>As users engage with the suggestions and provide feedback, Pathways adapts by learning from this input, continuously refining its recommendations to support the user's career journey as it evolves.</p>
                <p>You will begin with an onboarding assessment that determines the skills, work culture, and core values that are important to you in your ideal career path.</p>

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
                <p>To have a meaningful life and career, it is important to know what matters to you.</p>
                <p>This assessment helps you identify and prioritize what is essential to you at your core, work cultures that fit for you, and skills you enjoy using. This is a practical tool that will give you concrete information you can use to understand and tell your story.</p>
                <p>You will be sorting three categories of cards: Skills, Work Culture, and Core Values.</p>
                <p>Consider what is most important to you in an ideal work situation. Use this frame of reference when you are sorting the cards. You do not need to think about a specific job.</p>
                <p>You get to define what each word or phrase means to you. There is no need to follow the standard definition of a word.</p>
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
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
              <p className="font-circular font-bold text-4xl tracking-tight">The basic steps</p>
              <div className="flex flex-col space-y-4">
                <p>1. You will be sorting each card, one category at a time, into High, Medium, and Low areas to indicate the level of importance and/or enjoyment for you. The number of cards you put in each area is up to you.</p>
                <p>2. To move the cards around for each category, you can drag them or you can hover over the card and use the arrow menu.</p>
                <p>3. For each category, you'll take your High cards and narrow them down to your top 5.</p>
                <p>4. Next you'll rank order your top 5 from 1 to 5. If you are having trouble ranking the cards, focus on what energizes you. You will come back to these top 5 lists.</p>
                <p>5. At the end of the assessment you will have the opportunity to download a PDF of your results.</p>
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 3 && (
            <motion.div
              key="step3"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
            <div>
              <p className="font-circular font-bold text-4xl tracking-tight mb-2">Skills</p>
              <p className="font-circular font-medium text-xl tracking-tight "><span className="font-bold text-tertiaryBlue">Guiding Question: </span>What skills would I like to be using most at work?</p>
              <p className="font-circular font-medium text-xl tracking-tight"><span className="font-bold text-tertiaryBlue">Action: </span>Sort the skills cards according to the level of enjoyment for you. Don’t worry about your skill level.</p>

            </div>
              <div className="flex flex-row space-x-4">
                <div className="flex flex-col space-y-3 pt-4">
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    High
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Medium
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Low
                  </div>
                </div>
                <SkillsGrid 
                    skills={skills} 
                    onHighPrioritySkillsChange={(skills) => setStep3HighSkills(skills)}
                />
              </div>
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Instructions for Skills Sorting</DialogTitle>
                  <DialogDescription>
                    Take your time to sort each skill based on how much you enjoy using it. Don't worry about your current proficiency level - focus on what you'd like to be doing in your ideal role.
                    
                    Drag each card to the appropriate column (High, Medium, or Low) based on your level of enjoyment.
                  </DialogDescription>
                </DialogHeader>
                <Button variant="secondary" onClick={() => setDialogOpen(false)}>Got it</Button>
              </DialogContent>
            </Dialog>

              <div className="flex flex-row items-center w-full justify-between mt-8">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {/* {currentStep === 4 && (
            <motion.div
              key="step4"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <div>
                <p className="font-circular font-bold text-4xl tracking-tight mb-2">Select Your Top 5 Skills</p>
                <p className="font-circular font-medium text-xl tracking-tight">
                  <span className="font-bold text-tertiaryBlue">Action: </span>
                  From your high-priority skills, drag your top 5 into the slots below.
                </p>
              </div>

              <div className="flex flex-col space-y-8 w-full">
                <div className="flex flex-row flex-wrap gap-4 p-4 min-h-[100px] bg-gray-100 rounded-lg">
                  {highSkills.map((skill) => (
                    <div
                      key={skill}
                      className="px-4 py-2 bg-white rounded-lg shadow cursor-move"
                      draggable
                    >
                      {skill}
                    </div>
                  ))}
                </div>

                <div className="flex flex-row justify-between w-full">
                  {[1, 2, 3, 4, 5].map((slot) => (
                    <div
                      key={slot}
                      className="w-[180px] h-[100px] bg-gray-200 rounded-lg flex items-center justify-center"
                    >
                      {topFiveSkills[slot - 1] || `Slot ${slot}`}
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex flex-row items-center w-full justify-between mt-8">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                  {[1, 2, 3, 4, 5, 6, 7].map((step) => (
                    <button
                      key={step}
                      onClick={() => setCurrentStep(step)}
                      className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                        ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                    >
                      {step}
                    </button>
                  ))}
                </div>
                <Button 
                  variant="secondary" 
                  onClick={handleNext}
                  disabled={topFiveSkills.length !== 5}
                >
                  Next
                </Button>
              </div>
            </motion.div>
          )} */}

          {currentStep === 4 && (
            <motion.div
              key="step4"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
                <div>
                    <p className="font-circular font-bold text-4xl tracking-tight mb-2">Work culture</p>
                    <p className="font-circular font-medium text-xl tracking-tight "><span className="font-bold text-tertiaryBlue">Guiding Question: </span>What motivates me and helps me thrive in a work environment?</p>
                    <p className="font-circular font-medium text-xl tracking-tight"><span className="font-bold text-tertiaryBlue">Action: </span>Sort the work culture cards according to the level of importance for you.</p>

                </div>
                <div className="flex flex-row space-x-4">
                    <div className="flex flex-col space-y-3 pt-4">
                    <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                        High
                    </div>
                    <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                        Medium
                    </div>
                    <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                        Low
                    </div>
                    </div>
                    <SkillsGrid skills={workCulture} />
                </div>
                <div className="flex flex-row items-center w-full justify-between mt-8">
                    <Button variant="secondary" onClick={handleBack}>Back</Button>
                    <div className="flex flex-row space-x-2 items-center">
                        {[1, 2, 3, 4, 5, 6].map((step) => (
                            <button
                                key={step}
                                onClick={() => setCurrentStep(step)}
                                className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                    ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                            >
                                {step}
                            </button>
                        ))}
                    </div>
                    <Button variant="secondary" onClick={handleNext}>Next</Button>
                </div>
            </motion.div>
          )}

          {currentStep === 5 && (
            <motion.div
              key="step5"
              className="mx-auto flex-col justify-center space-y-6 items-start py-12 w-[92%]"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
                <div>
                    <p className="font-circular font-bold text-4xl tracking-tight mb-2">Core values</p>
                    <p className="font-circular font-medium text-xl tracking-tight "><span className="font-bold text-tertiaryBlue">Guiding Question: </span>What is integral to how I define my way of being and live my life?</p>
                    <p className="font-circular font-medium text-xl tracking-tight"><span className="font-bold text-tertiaryBlue">Action: </span>Sort the core values cards according to the level of importance for you. Consider what is essential to you.</p>
                </div>
              <div className="flex flex-row space-x-4">
                <div className="flex flex-col space-y-3 pt-4">
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    High
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Medium
                  </div>
                  <div className="w-[100px] h-[100px] rounded-2xl bg-gray-700 flex items-center justify-center text-white font-bold">
                    Low
                  </div>
                </div>
                <SkillsGrid skills={coreValues} />
              </div>
              <div className="flex flex-row items-center w-full justify-between mt-8">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleNext}>Next</Button>
              </div>
            </motion.div>
          )}

          {currentStep === 6 && (
            <motion.div
              key="step6"
              className="w-1/3 mx-auto flex-col justify-center space-y-6 items-start py-24"
              variants={fadeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              <p className="font-circular font-bold text-4xl tracking-tight">Anything else?</p>
              <div className="flex flex-col space-y-4 pb-24">
                <Textarea 
                  className="min-h-[250px]" 
                  placeholder="Share any additional thoughts, questions, or context about your career journey..."
                />
              </div>
              <div className="flex flex-row items-center w-full justify-between">
                <Button variant="secondary" onClick={handleBack}>Back</Button>
                <div className="flex flex-row space-x-2 items-center">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <button
                            key={step}
                            onClick={() => setCurrentStep(step)}
                            className={`w-[36px] h-[36px] flex items-center justify-center rounded-lg
                                ${currentStep === step ? 'bg-tertiaryBlue text-primaryBlue' : 'bg-secondaryBlue text-white'}`}
                        >
                            {step}
                        </button>
                    ))}
                </div>
                <Button variant="secondary" onClick={handleFinish}>Finish</Button>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>
    </div>
  );
}
