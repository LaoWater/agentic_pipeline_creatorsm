from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Colors
PRIMARY_COLOR = HexColor('#1a365d')  # Dark blue
ACCENT_COLOR = HexColor('#2b6cb0')   # Medium blue
TEXT_COLOR = HexColor('#2d3748')     # Dark gray
LIGHT_GRAY = HexColor('#718096')     # Light gray for secondary text

# Create custom styles
styles = getSampleStyleSheet()

# Name style - large and bold
name_style = ParagraphStyle(
    'NameStyle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=PRIMARY_COLOR,
    spaceAfter=2,
    alignment=TA_LEFT,
    fontName='Helvetica-Bold'
)

# Title/subtitle style
title_style = ParagraphStyle(
    'TitleStyle',
    parent=styles['Normal'],
    fontSize=12,
    textColor=ACCENT_COLOR,
    spaceAfter=8,
    fontName='Helvetica'
)

# Contact style
contact_style = ParagraphStyle(
    'ContactStyle',
    parent=styles['Normal'],
    fontSize=9,
    textColor=LIGHT_GRAY,
    spaceAfter=12,
    fontName='Helvetica'
)

# Section header style
section_style = ParagraphStyle(
    'SectionStyle',
    parent=styles['Heading2'],
    fontSize=12,
    textColor=PRIMARY_COLOR,
    spaceBefore=14,
    spaceAfter=6,
    fontName='Helvetica-Bold',
    borderPadding=(0, 0, 3, 0)
)

# Job title style
job_title_style = ParagraphStyle(
    'JobTitleStyle',
    parent=styles['Normal'],
    fontSize=11,
    textColor=PRIMARY_COLOR,
    spaceBefore=8,
    spaceAfter=1,
    fontName='Helvetica-Bold'
)

# Company style
company_style = ParagraphStyle(
    'CompanyStyle',
    parent=styles['Normal'],
    fontSize=10,
    textColor=ACCENT_COLOR,
    spaceAfter=4,
    fontName='Helvetica-Oblique'
)

# Normal text style
body_style = ParagraphStyle(
    'BodyStyle',
    parent=styles['Normal'],
    fontSize=9.5,
    textColor=TEXT_COLOR,
    spaceAfter=4,
    alignment=TA_JUSTIFY,
    fontName='Helvetica',
    leading=12
)

# Bullet style
bullet_style = ParagraphStyle(
    'BulletStyle',
    parent=styles['Normal'],
    fontSize=9.5,
    textColor=TEXT_COLOR,
    spaceAfter=3,
    leftIndent=12,
    bulletIndent=0,
    fontName='Helvetica',
    leading=12
)

# Skills style
skills_style = ParagraphStyle(
    'SkillsStyle',
    parent=styles['Normal'],
    fontSize=9,
    textColor=TEXT_COLOR,
    spaceAfter=2,
    fontName='Helvetica',
    leading=11
)

def create_cv():
    doc = SimpleDocTemplate(
        "Ioan_Criste_CV.pdf",
        pagesize=A4,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    story = []
    
    # Header
    story.append(Paragraph("IOAN \"N1ptic\" CRISTE", name_style))
    story.append(Paragraph("Full-Stack AI Engineer | Python Developer | Blockchain Specialist", title_style))
    story.append(Paragraph(
        'Cluj-Napoca, Romania  •  creators-multiverse.com  •  '
        '<link href="https://www.linkedin.com/in/ioan-criste/">LinkedIn</link>  •  '
        '<link href="https://github.com/N1ptic">GitHub</link>',
        contact_style
    ))
    
    # Divider
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=10))
    
    # Professional Summary
    story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
    
    summary_p1 = """Results-driven full-stack engineer with a proven track record of delivering AI-powered applications from conception to production. Experienced in building scalable content generation platforms, integrating cutting-edge AI models (Google Gemini, OpenAI), and maintaining stable production environments."""
    story.append(Paragraph(summary_p1, body_style))
    
    summary_p2 = """Strong background in blockchain infrastructure and high-performance trading systems. Skilled at iterative product improvement and cross-functional collaboration with research and marketing teams."""
    story.append(Paragraph(summary_p2, body_style))
    
    summary_p3 = """Working across both AI development and crypto has taught me an unforgettable lesson: client and market needs dictate the design of the tech stack and approach—not the other way around. I am committed to finding the framework that always best fulfills the client and product needs."""
    story.append(Paragraph(summary_p3, body_style))
    
    # Technical Skills
    story.append(Paragraph("TECHNICAL SKILLS", section_style))
    
    skills_data = [
        ["<b>Languages:</b>", "Python, Rust, TypeScript, JavaScript, SQL"],
        ["<b>Frontend:</b>", "React, Vite, shadcn/ui, TailwindCSS, React Query, React Router, Framer Motion"],
        ["<b>Backend:</b>", "FastAPI, Node.js, Express, LangChain, Pydantic, AsyncIO"],
        ["<b>AI/ML:</b>", "Google Gemini AI, OpenAI API, Google Imagen, Multi-Agent Systems, Prompt Engineering"],
        ["<b>Database:</b>", "PostgreSQL, Supabase, Row Level Security, Database Functions &amp; Triggers"],
        ["<b>Cloud &amp; DevOps:</b>", "Google Cloud Platform (App Engine, Cloud Storage, Cloud Run), Docker, Serverless"],
        ["<b>Blockchain:</b>", "Crypto Trading Bots, DeFi Protocol Integration, Smart Contract Interaction"],
        ["<b>Tools:</b>", "Git, Playwright, ESLint, Stripe API, REST APIs, Deno Runtime"],
    ]
    
    for skill_row in skills_data:
        story.append(Paragraph(f"{skill_row[0]} {skill_row[1]}", skills_style))
    
    # Professional Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
    
    # Job 1: CreatorsM
    story.append(Paragraph("Full-Stack AI Engineer | 2022 – Present", job_title_style))
    story.append(Paragraph("Creators Multiverse | Cluj-Napoca, Romania", company_style))
    
    story.append(Paragraph("Critical part in end-to-end development and production launch of AI-powered multi-platform social media content generation platform, delivering a scalable solution serving multiple brands across 5+ social media platforms.", body_style))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Product Development &amp; Launch:</b>", bullet_style))
    bullets_1 = [
        "• Delivered production-ready platform from conception to launch, coordinating with research team to integrate cutting-edge AI models (Google Gemini, OpenAI) and ensuring brand voice consistency across all generated content",
        "• Established dual-ecosystem AI strategy offering users choice between Google Gemini and OpenAI providers, transforming technical deployment challenges into unique market differentiator",
        "• Launched comprehensive content creation solution supporting Instagram, LinkedIn, Twitter, Facebook, TikTok, and YouTube with platform-specific optimizations",
    ]
    for b in bullets_1:
        story.append(Paragraph(b, bullet_style))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Platform Stability &amp; Performance:</b>", bullet_style))
    bullets_2 = [
        "• Maintained stable production environment with robust error handling, ensuring reliable content generation and minimizing service disruptions through comprehensive testing and monitoring",
        "• Optimized content generation pipeline reducing processing time through parallel execution architecture, improving user experience with faster delivery of AI-generated posts and media assets",
        "• Implemented scalable cloud infrastructure on Google Cloud Platform with automatic scaling capabilities",
    ]
    for b in bullets_2:
        story.append(Paragraph(b, bullet_style))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Business Impact &amp; Revenue:</b>", bullet_style))
    bullets_3 = [
        "• Integrated Stripe payment system enabling multiple pricing tiers with automated credit management, establishing revenue stream and supporting business growth",
        "• Built influencer partnership program with referral tracking and commission system, creating channel for organic user acquisition",
        "• Developed multi-company support allowing agencies to manage multiple brand identities within single account",
    ]
    for b in bullets_3:
        story.append(Paragraph(b, bullet_style))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Technical Stack:</b> React/TypeScript frontend, Python/FastAPI backend, PostgreSQL with Supabase, Google Cloud Platform, Stripe integrations, Google AI &amp; OpenAI APIs", bullet_style))
    
    # Job 2: Freelance
    story.append(Spacer(1, 8))
    story.append(Paragraph("Blockchain Infrastructure Engineer &amp; Automation Specialist | 2015 – 2022", job_title_style))
    story.append(Paragraph("Independent Contractor (Freelance) | Remote", company_style))
    
    story.append(Paragraph("Delivered high-performance cryptocurrency trading systems and automation solutions for private clients, focusing on reliable execution, risk management, and measurable ROI in volatile market conditions.", body_style))
    
    bullets_freelance = [
        "• Developed and deployed automated trading systems for multiple clients, achieving consistent performance in live trading environments across major cryptocurrency exchanges with 99%+ uptime",
        "• Delivered blockchain infrastructure solutions reducing manual operational overhead for clients managing DeFi protocols and Web3 applications, enabling 24/7 monitoring and automated response to on-chain events",
        "• Built advanced automation frameworks streamlining repetitive workflows and data extraction processes, reducing processing time from hours to minutes and eliminating manual errors",
        "• Ensured system reliability through comprehensive error handling, failover mechanisms, and monitoring systems for mission-critical trading infrastructure",
        "• Maintained strict security and confidentiality for proprietary trading algorithms and client data, implementing secure API key management",
    ]
    for b in bullets_freelance:
        story.append(Paragraph(b, bullet_style))
    
    story.append(Paragraph("<b>Technical Environment:</b> Rust, Node.js, TypeScript, Blockchain APIs, Real-time data processing, Cloud infrastructure, Webhooks, REST APIs, Python", bullet_style))
    
    # Job 3: SkillLab
    story.append(Spacer(1, 8))
    story.append(Paragraph("Python &amp; Web Development Intern | 2012 – 2014", job_title_style))
    story.append(Paragraph("CodersLab | Cluj-Napoca, Romania", company_style))
    
    bullets_intern = [
        "• Completed intensive one-year+ training program in Python programming and modern web development practices",
        "• Gained hands-on experience building web applications and working with databases",
        "• Developed foundational skills in software engineering principles and collaborative development workflows",
    ]
    for b in bullets_intern:
        story.append(Paragraph(b, bullet_style))
    
    # Education
    story.append(Paragraph("EDUCATION", section_style))
    story.append(Paragraph("Bachelor of Computer Science", job_title_style))
    story.append(Paragraph("Technical University of Cluj-Napoca | Cluj-Napoca, Romania | 2012", company_style))
    
    # Key Projects
    story.append(Paragraph("KEY PROJECTS", section_style))
    
    story.append(Paragraph("<b>CreatorsM</b> – AI Content Generation Platform", bullet_style))
    story.append(Paragraph("Production SaaS platform enabling brands to generate platform-optimized social media content across Instagram, LinkedIn, Twitter, Facebook, TikTok, and YouTube using AI. Full-stack development with dual AI provider integration (Google Gemini + OpenAI), cloud deployment on GCP, and Stripe payment processing.", body_style))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Cryptocurrency Trading Systems</b> – Private Client Work", bullet_style))
    story.append(Paragraph("Automated trading infrastructure for cryptocurrency markets with multi-exchange support and risk management. Delivered reliable 24/7 trading automation with 99%+ uptime for multiple clients, enabling systematic trading strategies in volatile markets.", body_style))
    
    # Certifications
    story.append(Paragraph("CERTIFICATIONS &amp; PROFESSIONAL DEVELOPMENT", section_style))
    certs = [
        "• Advanced Prompt Engineering for AI Systems",
        "• Google Cloud Platform Architecture",
        "• Blockchain Development &amp; Smart Contracts",
        "• FastAPI &amp; Modern Python Development",
        "• React &amp; TypeScript Best Practices",
    ]
    for c in certs:
        story.append(Paragraph(c, bullet_style))
    
    # Languages
    story.append(Paragraph("LANGUAGES", section_style))
    story.append(Paragraph("<b>English:</b> Professional Working Proficiency  |  <b>Romanian:</b> Native", skills_style))
    
    # Build PDF
    doc.build(story)
    print("CV created successfully!")

if __name__ == "__main__":
    create_cv()