"use client";

import Cookies from "js-cookie";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { createClient } from '@supabase/supabase-js'
import { Button } from "@/components/ui/button";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { useMediaQuery } from "@/utils/use-media-query";

import {
  Command,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Drawer,
  DrawerContent,
  DrawerTrigger,
} from "@/components/ui/drawer"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

type Option = {
  value: string
  label: string
}

const options: Option[] = [
  {
    value: "is_investor",
    label: "I'm an investor",
  },
  {
    value: "is_new_builder",
    label: "I’m new to building ideas",
  },
  {
    value: "is_experienced_builder",
    label: "I've built stuff in the past",
  },
  {
    value: "is_partner",
    label: "I'm a prospective partner",
  },
  {
    value: "is_vc_founder",
    label: "I've raised money",
  },
]

const EmailCapture = () => {
  const [open, setOpen] = useState(false);
  const [alertVisible, setAlertVisible] = useState(false);
  const [alertMessage, setAlertMessage] = useState("");
  const [alertVariant, setAlertVariant] = useState<"default" | "destructive" | null>("default");

  const isDesktop = useMediaQuery("(min-width: 768px)");

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_API_KEY!;
  const supabase = createClient(supabaseUrl, supabaseKey);

  const [email, setEmail] = useState("");
  const [emailError, setEmailError] = useState(false);
  const [isEmailSubmitted, setIsEmailSubmitted] = useState(false);
  const router = useRouter();

  const validateEmail = (email: string) => {
    return /\S+@\S+\.\S+/.test(email);
  };

  const handleSubmit = async () => {
    if (!validateEmail(email)) {
      setEmailError(true);
      setAlertMessage("We were unable to submit your email address. Please try again.");
      setAlertVariant("destructive");
      setAlertVisible(true);
      setTimeout(() => setAlertVisible(false), 3000);
      return;
    }
    setEmailError(false);
    try {
      const { data, error } = await supabase
        .from('emails')
        .insert([
          { email: email }
        ]);
      if (error) throw error;
      setAlertMessage("Success! You've been added to the Startup Exchange mailing list.");
      setAlertVariant("default");
      setIsEmailSubmitted(true);
      setOpen(true);
    } catch (error) {
      console.error('Error inserting data: ', error);
      setEmailError(true);
      setAlertMessage("Error: We were unable to submit your email address. Please try again.");
      setAlertVariant("destructive");
      setAlertVisible(true);
      setTimeout(() => setAlertVisible(false), 3000);
    }
  };

  const handleOptionSelect = async (value: string) => {
    try {
      const { data, error } = await supabase
        .from('emails')
        .update({ [value]: true })
        .eq('email', email);
      if (error) throw error;
      console.log('Update success:', data);
      setEmail("");
      setOpen(false);
      setAlertVisible(true);
      setTimeout(() => setAlertVisible(false), 3000);
    } catch (error) {
      console.error('Error updating data: ', error);
    }
  };

  // const checkEmail = async (email: string) => {
  //   try {
  //     const { data, error } = await supabase
  //       .from('emails')
  //       .select()
  //       .eq('email', email);
  //     if (error) throw error;
  //     console.log('Email check:', data);
  //     return data;
  //   } catch (error) {
  //     console.error('Error checking email:', error);
  //   }
  // };

  // const testUpdate = async () => {
  //   try {
  //     const { data, error } = await supabase
  //       .from('emails')
  //       // .select()
  //       .update({ is_investor : true })
  //       .eq('email', 'thegamingprotocol06@gmail.com')
  //       .select();
  //     if (error) throw error;
  //     console.log('Test update:', data);
  //   } catch (error) {
  //     console.error('Error during test update:', error);
  //   }
  // };

  function ComboBoxResponsive() {
    const commonProps = {
      onClick: (evt: React.MouseEvent<HTMLButtonElement>) => {
        evt.preventDefault();
        handleSubmit();
      },
      className: "md:px-6 px-3 md:mb-0 mb-2",
      variant: "secondary" as const,
    };

    if (isDesktop) {
      return (
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button {...commonProps}>Subscribe for updates</Button>
          </PopoverTrigger>
          <PopoverContent className="p-0" align="start">
            <OptionList />
          </PopoverContent>
        </Popover>
      )
    }

    return (
      <Drawer open={open} onOpenChange={setOpen}>
        <DrawerTrigger asChild>
          <div className="w-full">
            <Button {...commonProps}>Get connected</Button>
          </div>
        </DrawerTrigger>
        <DrawerContent>
          <div className="mt-4">
            <OptionList />
          </div>
        </DrawerContent>
      </Drawer>
    )
  }

  function OptionList() {
    return (
      <Command>
        <CommandList>
          <CommandGroup>
            {options.map((option) => (
              <CommandItem
                key={option.value}
                value={option.value}
                onSelect={() => {
                  handleOptionSelect(option.value);
                  // setOpen(false);
                }}
              >
                {option.label}
              </CommandItem>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    )
  }

  return (
    <>
      {alertVisible && (
        <div className="fixed bottom-5 right-5 animate-slide-up">
          <Alert variant={alertVariant}>
              <div className="flex flex-col">
                <AlertTitle>{alertVariant === "destructive" ? "Error!" : "Success!"}</AlertTitle>
                <AlertDescription>
                  {alertMessage}
                </AlertDescription>
              </div>
          </Alert>
        </div>
      )}
      <div
        className={`flex md:flex-row flex-col px-2 mb-8 mt-4 box-content w-full max-w-[485px] items-center rounded-[16px] bg-secondaryBlue  ${emailError ? "border border-red-500" : ""}`}
      >
        <input
          key={isEmailSubmitted ? "submitted" : "not-submitted"}
          className={`fg-dark-24 w-full bg-transparent py-4 md:pl-2 pl-1.5 md:pr-4 pr-3.5 font-circular md:text-base text-sm leading-[1.5rem] outline-none text-[#CECECE] drop-shadow-xl ${emailError ? "border-red-500 placeholder-red-500" : ""}`}
          type="text"
          onChange={(evt) => {
            setEmail(evt.target.value);
            if (emailError) setEmailError(false);
          }}
          maxLength={80}
          required={true}
          placeholder={"Enter your email address"}
        />
        <ComboBoxResponsive/>
        {/* <Button
          variant="secondary" 
          className="px-8" 
          onClick={() => checkEmail('cartercote06@gmail.com')}>
          checkEmail
        </Button> */}
        {/* <Button
          variant="secondary" 
          className="px-8" 
          onClick={() => testUpdate()}>
          testUpdate
        </Button> */}
      </div>
    </>
  );
};

export default EmailCapture;