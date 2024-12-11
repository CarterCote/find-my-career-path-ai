## Running the project locally

To run the backend and view the chatbot in action, follow the instructions:

1. Start the FastAPI server:
```bash
cd backend
uvicorn src.main:app --reload
```

1. In a separate terminal, start the CLI interface:
```bash
cd backend
python src/temp_chat_cli.py
```

## Viewing the frontend with card-sorting activity

First, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can view the card-sorting activity at the /try endpoint. Clicking through the card activity and clicking "Finish" will allow you to save your profile.

Given that there's no way to track your user ID on the frontend, you can use the chatbot by accessing user IDs like 79 or 80.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!