import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const data = await request.json();
    console.log('Received data in API route:', data);
    
    // Generate a random session ID
    const sessionId = `session-${Math.floor(Math.random() * 9000000) + 1000}`;
    
    // Format the data to exactly match the Python backend's expectations
    const formattedData = {
      user_session_id: sessionId,
      skills: data.skills,
      work_culture: data.work_culture,
      core_values: data.core_values,
      additional_interests: data.additional_interests || ""
    };

    console.log('Sending to backend:', formattedData);

    const response = await fetch('http://127.0.0.1:8000/users/preferences', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formattedData)
    });

    const responseText = await response.text();
    console.log('Backend response text:', responseText);

    if (!response.ok) {
      throw new Error(`Failed to save to backend: ${responseText}`);
    }

    return NextResponse.json({ success: true, sessionId });
    
  } catch (error) {
    console.error('Error in profile API route:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: 'Failed to save profile', details: errorMessage },
      { status: 500 }
    );
  }
}