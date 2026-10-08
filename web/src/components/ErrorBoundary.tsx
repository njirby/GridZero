import { Component, type ReactNode } from "react";

// Keeps one bad render (e.g. an unexpected payload shape) from unmounting the whole app.
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return <div className="error-banner" role="alert" style={{ position: "static", margin: 16 }}>
      Something went wrong rendering the UI: {this.state.error.message}
      <button onClick={() => this.setState({ error: null })}>Retry</button>
    </div>;
  }
}
