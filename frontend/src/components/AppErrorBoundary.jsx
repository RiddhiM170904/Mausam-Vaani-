import React from "react";
import ServiceUnavailable from "./ServiceUnavailable";

export default class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("Unhandled UI error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <ServiceUnavailable
          title="Service Temporarily Unavailable"
          message="We are not servicing at this time. Sorry for the inconvenience."
        />
      );
    }

    return this.props.children;
  }
}
