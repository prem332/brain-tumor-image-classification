import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Brain Tumor AI',
  description: 'VGG16 MRI Classification',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body style={{ background: '#0a0f1e', minHeight: '100vh', margin: 0 }}>
        {children}
      </body>
    </html>
  )
}
