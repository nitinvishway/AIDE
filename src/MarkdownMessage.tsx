import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownMessageProps {
  content: string;
}

export const MarkdownMessage: React.FC<MarkdownMessageProps> = ({ content }) => {
  return (
    <div className="md-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const codeText = String(children).replace(/\n$/, "");
            const isInline = !match && !codeText.includes("\n");

            if (isInline) {
              return (
                <code className="md-inline-code" {...props}>
                  {children}
                </code>
              );
            }

            return <CodeBlock language={match ? match[1] : ""} code={codeText} />;
          },
          p({ children }) {
            return <p className="md-p">{children}</p>;
          },
          ul({ children }) {
            return <ul className="md-ul">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="md-ol">{children}</ol>;
          },
          li({ children }) {
            return <li className="md-li">{children}</li>;
          },
          h1({ children }) {
            return <h4 className="md-h">{children}</h4>;
          },
          h2({ children }) {
            return <h4 className="md-h">{children}</h4>;
          },
          h3({ children }) {
            return <h5 className="md-h">{children}</h5>;
          },
          h4({ children }) {
            return <h5 className="md-h">{children}</h5>;
          },
          blockquote({ children }) {
            return <blockquote className="md-blockquote">{children}</blockquote>;
          },
          table({ children }) {
            return (
              <div className="md-table-wrapper">
                <table className="md-table">{children}</table>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="md-code-block">
      <div className="md-code-header">
        <span className="md-code-lang">{language || "code"}</span>
        <button className="md-code-copy-btn" onClick={handleCopy} title="Copy code">
          {copied ? "Copied ✓" : "Copy"}
        </button>
      </div>
      <pre className="md-code-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
}
