import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const data = await request.json();
    
    // Send data to your backend
    const response = await fetch('http://localhost:8000/users/preferences', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: "test-session", // You can generate this dynamically
        core_values: data.core_values.slice(0, 10),
        work_culture: data.work_culture.slice(0, 10),
        skills: data.skills.slice(0, 10),
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