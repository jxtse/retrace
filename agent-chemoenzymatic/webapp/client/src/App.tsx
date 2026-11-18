import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RunPage } from "./pages/RunPage";

const qc = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <RunPage />
    </QueryClientProvider>
  );
}
