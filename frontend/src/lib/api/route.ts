import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const data = await request.json();
    
    // Send data to your backend
    const response = await fetch('YOUR_BACKEND_URL/user_profiles', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        skills: data.skills,
        work_culture: data.work_culture,
        core_values: data.core_values,
        additional_interests: data.additional_interests
      })
    });

    if (!response.ok) {
      throw new Error('Failed to save to backend');
    }

    return NextResponse.json({ success: true });
    
  } catch (error) {
    console.error('Error in profile API route:', error);
    return NextResponse.json(
      { error: 'Failed to save profile' },
      { status: 500 }
    );
  }
}