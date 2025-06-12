"""
Category Configuration for TheBigEverythingPromptLibrary
Defines the hierarchical categorization system for prompts
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class SubCategory:
    """Represents a subcategory within a main category"""
    id: str
    name: str
    description: str
    keywords: List[str]  # Keywords to help auto-categorize
    icon: Optional[str] = None  # Optional icon identifier

@dataclass
class Category:
    """Represents a main category"""
    id: str
    name: str
    description: str
    subcategories: List[SubCategory]
    keywords: List[str]
    icon: Optional[str] = None
    color: Optional[str] = None  # For UI theming

# Define the comprehensive categorization system
CATEGORIES = {
    "coding_development": Category(
        id="coding_development",
        name="Coding & Development",
        description="Programming assistants, code generators, and development tools",
        keywords=["code", "programming", "development", "software", "api", "debug", "compile"],
        color="#2563eb",
        icon="code",
        subcategories=[
            SubCategory(
                id="general_coding",
                name="General Coding",
                description="Multi-language coding assistants",
                keywords=["code", "programming", "developer", "software"]
            ),
            SubCategory(
                id="web_development",
                name="Web Development",
                description="Frontend, backend, and full-stack tools",
                keywords=["web", "html", "css", "javascript", "react", "vue", "node"]
            ),
            SubCategory(
                id="mobile_development",
                name="Mobile Development",
                description="iOS, Android, and cross-platform development",
                keywords=["mobile", "ios", "android", "flutter", "react native"]
            ),
            SubCategory(
                id="game_development",
                name="Game Development",
                description="Game engines and game programming",
                keywords=["game", "unity", "unreal", "godot", "gamedev"]
            ),
            SubCategory(
                id="devops_infrastructure",
                name="DevOps & Infrastructure",
                description="CI/CD, cloud, and infrastructure tools",
                keywords=["devops", "docker", "kubernetes", "aws", "azure", "ci/cd"]
            ),
            SubCategory(
                id="database_data",
                name="Database & Data",
                description="SQL, NoSQL, and data management",
                keywords=["database", "sql", "nosql", "mongodb", "postgresql"]
            )
        ]
    ),
    
    "security_hacking": Category(
        id="security_hacking",
        name="Security & Ethical Hacking",
        description="Cybersecurity, penetration testing, and security research",
        keywords=["security", "hacking", "ctf", "penetration", "vulnerability", "exploit"],
        color="#dc2626",
        icon="shield",
        subcategories=[
            SubCategory(
                id="ethical_hacking",
                name="Ethical Hacking",
                description="White hat hacking and security testing",
                keywords=["ethical", "white hat", "pentest", "security testing"]
            ),
            SubCategory(
                id="ctf_challenges",
                name="CTF & Challenges",
                description="Capture The Flag and security puzzles",
                keywords=["ctf", "capture the flag", "challenge", "puzzle"]
            ),
            SubCategory(
                id="security_analysis",
                name="Security Analysis",
                description="Vulnerability assessment and security auditing",
                keywords=["vulnerability", "audit", "analysis", "assessment"]
            ),
            SubCategory(
                id="prompt_security",
                name="Prompt Security",
                description="Prompt injection defense and jailbreak prevention",
                keywords=["prompt injection", "jailbreak", "defense", "protection"]
            )
        ]
    ),
    
    "creative_art": Category(
        id="creative_art",
        name="Creative & Art",
        description="Image generation, design tools, and artistic creation",
        keywords=["art", "design", "creative", "image", "visual", "graphic", "draw"],
        color="#8b5cf6",
        icon="palette",
        subcategories=[
            SubCategory(
                id="image_generation",
                name="Image Generation",
                description="AI image and art generators",
                keywords=["image", "generate", "dall-e", "midjourney", "stable diffusion"]
            ),
            SubCategory(
                id="graphic_design",
                name="Graphic Design",
                description="Design tools and templates",
                keywords=["design", "graphic", "logo", "branding", "visual"]
            ),
            SubCategory(
                id="ui_ux_design",
                name="UI/UX Design",
                description="Interface and experience design",
                keywords=["ui", "ux", "interface", "user experience", "wireframe"]
            ),
            SubCategory(
                id="style_fashion",
                name="Style & Fashion",
                description="Fashion, style, and aesthetic tools",
                keywords=["fashion", "style", "aesthetic", "outfit", "color"]
            )
        ]
    ),
    
    "writing_content": Category(
        id="writing_content",
        name="Writing & Content",
        description="Writing assistants, editors, and content generation",
        keywords=["write", "writing", "content", "text", "editor", "author"],
        color="#059669",
        icon="edit",
        subcategories=[
            SubCategory(
                id="general_writing",
                name="General Writing",
                description="All-purpose writing assistants",
                keywords=["write", "writing", "text", "compose"]
            ),
            SubCategory(
                id="creative_writing",
                name="Creative Writing",
                description="Fiction, poetry, and storytelling",
                keywords=["story", "fiction", "poetry", "creative", "narrative"]
            ),
            SubCategory(
                id="technical_writing",
                name="Technical Writing",
                description="Documentation and technical content",
                keywords=["technical", "documentation", "manual", "guide"]
            ),
            SubCategory(
                id="academic_writing",
                name="Academic Writing",
                description="Research papers and academic content",
                keywords=["academic", "research", "paper", "thesis", "dissertation"]
            ),
            SubCategory(
                id="copywriting",
                name="Copywriting",
                description="Marketing and sales copy",
                keywords=["copy", "marketing", "sales", "advertising", "seo"]
            ),
            SubCategory(
                id="prompt_engineering",
                name="Prompt Engineering",
                description="Prompt creation and optimization",
                keywords=["prompt", "engineering", "optimize", "craft"]
            )
        ]
    ),
    
    "business_professional": Category(
        id="business_professional",
        name="Business & Professional",
        description="Business tools, productivity, and professional services",
        keywords=["business", "professional", "work", "office", "productivity"],
        color="#0891b2",
        icon="briefcase",
        subcategories=[
            SubCategory(
                id="productivity_tools",
                name="Productivity Tools",
                description="Task management and efficiency tools",
                keywords=["productivity", "task", "management", "efficiency"]
            ),
            SubCategory(
                id="hr_recruitment",
                name="HR & Recruitment",
                description="Human resources and hiring tools",
                keywords=["hr", "human resources", "recruitment", "hiring", "resume"]
            ),
            SubCategory(
                id="finance_accounting",
                name="Finance & Accounting",
                description="Financial analysis and accounting tools",
                keywords=["finance", "accounting", "tax", "budget", "investment"]
            ),
            SubCategory(
                id="marketing_sales",
                name="Marketing & Sales",
                description="Marketing strategies and sales tools",
                keywords=["marketing", "sales", "advertising", "campaign", "lead"]
            ),
            SubCategory(
                id="project_management",
                name="Project Management",
                description="Project planning and team coordination",
                keywords=["project", "management", "agile", "scrum", "planning"]
            ),
            SubCategory(
                id="legal_compliance",
                name="Legal & Compliance",
                description="Legal documents and compliance tools",
                keywords=["legal", "law", "compliance", "contract", "policy"]
            )
        ]
    ),
    
    "education_learning": Category(
        id="education_learning",
        name="Education & Learning",
        description="Educational tools, tutors, and learning assistants",
        keywords=["education", "learning", "teach", "tutor", "study", "course"],
        color="#f59e0b",
        icon="graduation-cap",
        subcategories=[
            SubCategory(
                id="subject_tutors",
                name="Subject Tutors",
                description="Math, science, language tutors",
                keywords=["tutor", "math", "science", "language", "subject"]
            ),
            SubCategory(
                id="language_learning",
                name="Language Learning",
                description="Language practice and translation",
                keywords=["language", "translation", "vocabulary", "grammar"]
            ),
            SubCategory(
                id="homework_help",
                name="Homework Help",
                description="Assignment and homework assistance",
                keywords=["homework", "assignment", "help", "study"]
            ),
            SubCategory(
                id="exam_prep",
                name="Exam Preparation",
                description="Test prep and study guides",
                keywords=["exam", "test", "prep", "sat", "gre", "preparation"]
            ),
            SubCategory(
                id="skill_development",
                name="Skill Development",
                description="Professional and personal skill building",
                keywords=["skill", "development", "training", "course"]
            )
        ]
    ),
    
    "roleplay_characters": Category(
        id="roleplay_characters",
        name="Roleplay & Characters",
        description="Character roleplay, personas, and interactive personalities",
        keywords=["roleplay", "character", "persona", "personality", "companion"],
        color="#ec4899",
        icon="masks-theater",
        subcategories=[
            SubCategory(
                id="fictional_characters",
                name="Fictional Characters",
                description="Characters from media and original creations",
                keywords=["character", "fictional", "anime", "movie", "book"]
            ),
            SubCategory(
                id="historical_figures",
                name="Historical Figures",
                description="Historical personalities and figures",
                keywords=["historical", "history", "figure", "personality"]
            ),
            SubCategory(
                id="companions_friends",
                name="Companions & Friends",
                description="Virtual companions and friendly bots",
                keywords=["companion", "friend", "buddy", "pal", "mate"]
            ),
            SubCategory(
                id="romantic_roleplay",
                name="Romantic Roleplay",
                description="Dating sims and romantic companions",
                keywords=["romantic", "dating", "girlfriend", "boyfriend", "romance"]
            ),
            SubCategory(
                id="professional_personas",
                name="Professional Personas",
                description="Professional role simulations",
                keywords=["professional", "expert", "consultant", "advisor"]
            )
        ]
    ),
    
    "games_entertainment": Category(
        id="games_entertainment",
        name="Games & Entertainment",
        description="Interactive games, puzzles, and entertainment",
        keywords=["game", "play", "fun", "entertainment", "puzzle", "challenge"],
        color="#6366f1",
        icon="gamepad",
        subcategories=[
            SubCategory(
                id="text_adventures",
                name="Text Adventures",
                description="Interactive fiction and text-based games",
                keywords=["adventure", "text", "interactive fiction", "story game"]
            ),
            SubCategory(
                id="rpg_games",
                name="RPG Games",
                description="Role-playing games and character adventures",
                keywords=["rpg", "role playing", "character", "quest", "adventure"]
            ),
            SubCategory(
                id="puzzle_games",
                name="Puzzle Games",
                description="Logic puzzles and brain teasers",
                keywords=["puzzle", "logic", "brain teaser", "riddle"]
            ),
            SubCategory(
                id="trivia_quiz",
                name="Trivia & Quiz",
                description="Knowledge games and quizzes",
                keywords=["trivia", "quiz", "knowledge", "question"]
            ),
            SubCategory(
                id="creative_games",
                name="Creative Games",
                description="Creative and imagination-based games",
                keywords=["creative", "imagination", "story", "create"]
            )
        ]
    ),
    
    "lifestyle_personal": Category(
        id="lifestyle_personal",
        name="Lifestyle & Personal",
        description="Personal development, health, and lifestyle tools",
        keywords=["lifestyle", "personal", "health", "fitness", "life"],
        color="#10b981",
        icon="heart",
        subcategories=[
            SubCategory(
                id="health_fitness",
                name="Health & Fitness",
                description="Exercise, nutrition, and wellness",
                keywords=["health", "fitness", "exercise", "nutrition", "wellness"]
            ),
            SubCategory(
                id="cooking_food",
                name="Cooking & Food",
                description="Recipes and culinary assistance",
                keywords=["cooking", "recipe", "food", "cuisine", "chef"]
            ),
            SubCategory(
                id="fashion_style",
                name="Fashion & Style",
                description="Personal style and fashion advice",
                keywords=["fashion", "style", "outfit", "clothing", "wardrobe"]
            ),
            SubCategory(
                id="relationships",
                name="Relationships",
                description="Dating, relationship advice, and social skills",
                keywords=["relationship", "dating", "social", "communication"]
            ),
            SubCategory(
                id="mental_wellness",
                name="Mental Wellness",
                description="Mental health and emotional support",
                keywords=["mental", "wellness", "emotional", "therapy", "support"]
            ),
            SubCategory(
                id="hobbies_interests",
                name="Hobbies & Interests",
                description="Hobby-specific tools and guides",
                keywords=["hobby", "interest", "craft", "collection"]
            )
        ]
    ),
    
    "data_analysis": Category(
        id="data_analysis",
        name="Data & Analysis",
        description="Data analysis, research, and information processing",
        keywords=["data", "analysis", "research", "analytics", "statistics"],
        color="#7c3aed",
        icon="chart-bar",
        subcategories=[
            SubCategory(
                id="data_analysis_tools",
                name="Data Analysis Tools",
                description="Statistical and data analysis",
                keywords=["statistics", "analysis", "data", "analytics"]
            ),
            SubCategory(
                id="research_tools",
                name="Research Tools",
                description="Academic and market research",
                keywords=["research", "study", "survey", "academic"]
            ),
            SubCategory(
                id="summarization",
                name="Summarization",
                description="Content summarization and extraction",
                keywords=["summary", "summarize", "extract", "digest"]
            ),
            SubCategory(
                id="information_extraction",
                name="Information Extraction",
                description="Data extraction and parsing",
                keywords=["extract", "parse", "scrape", "information"]
            )
        ]
    ),
    
    "specialized_tools": Category(
        id="specialized_tools",
        name="Specialized Tools",
        description="Industry-specific and specialized professional tools",
        keywords=["specialized", "industry", "specific", "professional", "niche"],
        color="#f97316",
        icon="tools",
        subcategories=[
            SubCategory(
                id="medical_health",
                name="Medical & Healthcare",
                description="Medical information and healthcare tools",
                keywords=["medical", "healthcare", "doctor", "patient", "diagnosis"]
            ),
            SubCategory(
                id="scientific_research",
                name="Scientific Research",
                description="Scientific analysis and research tools",
                keywords=["science", "research", "laboratory", "experiment"]
            ),
            SubCategory(
                id="engineering_technical",
                name="Engineering & Technical",
                description="Engineering calculations and technical tools",
                keywords=["engineering", "technical", "calculation", "design"]
            ),
            SubCategory(
                id="real_estate",
                name="Real Estate",
                description="Property and real estate tools",
                keywords=["real estate", "property", "housing", "rental"]
            ),
            SubCategory(
                id="travel_tourism",
                name="Travel & Tourism",
                description="Travel planning and tourism guides",
                keywords=["travel", "tourism", "vacation", "trip", "guide"]
            )
        ]
    ),
    
    "spiritual_philosophical": Category(
        id="spiritual_philosophical",
        name="Spiritual & Philosophical",
        description="Spiritual guidance, philosophy, and mindfulness",
        keywords=["spiritual", "philosophy", "religion", "mindfulness", "meditation"],
        color="#9333ea",
        icon="yin-yang",
        subcategories=[
            SubCategory(
                id="religious_guidance",
                name="Religious Guidance",
                description="Religious texts and spiritual advice",
                keywords=["religion", "faith", "prayer", "scripture", "holy"]
            ),
            SubCategory(
                id="philosophy",
                name="Philosophy",
                description="Philosophical discussions and thought",
                keywords=["philosophy", "ethics", "morality", "wisdom"]
            ),
            SubCategory(
                id="mindfulness_meditation",
                name="Mindfulness & Meditation",
                description="Meditation guides and mindfulness practices",
                keywords=["mindfulness", "meditation", "zen", "calm", "peace"]
            ),
            SubCategory(
                id="astrology_mysticism",
                name="Astrology & Mysticism",
                description="Astrology, tarot, and mystical practices",
                keywords=["astrology", "tarot", "mystical", "zodiac", "fortune"]
            )
        ]
    ),
    
    "meta_tools": Category(
        id="meta_tools",
        name="Meta & GPT Tools",
        description="Tools for creating, managing, and protecting prompts",
        keywords=["meta", "gpt", "prompt", "tool", "builder", "protection"],
        color="#64748b",
        icon="cog",
        subcategories=[
            SubCategory(
                id="prompt_builders",
                name="Prompt Builders",
                description="Tools for creating and optimizing prompts",
                keywords=["builder", "creator", "generator", "optimizer"]
            ),
            SubCategory(
                id="gpt_management",
                name="GPT Management",
                description="GPT marketplace and management tools",
                keywords=["management", "marketplace", "store", "catalog"]
            ),
            SubCategory(
                id="security_testing",
                name="Security Testing",
                description="Prompt injection and security testing",
                keywords=["security", "injection", "testing", "vulnerability"]
            ),
            SubCategory(
                id="prompt_protection",
                name="Prompt Protection",
                description="Tools to protect prompts from extraction",
                keywords=["protection", "guard", "shield", "defense"]
            )
        ]
    )
}

# Helper functions for categorization
def get_category_by_id(category_id: str) -> Optional[Category]:
    """Get a category by its ID"""
    return CATEGORIES.get(category_id)

def get_subcategory_by_id(category_id: str, subcategory_id: str) -> Optional[SubCategory]:
    """Get a subcategory by its ID"""
    category = get_category_by_id(category_id)
    if category:
        for subcat in category.subcategories:
            if subcat.id == subcategory_id:
                return subcat
    return None

def suggest_category(text: str) -> Optional[Tuple[str, str]]:
    """Suggest a category and subcategory based on text content"""
    text_lower = text.lower()
    best_match = None
    highest_score = 0
    
    for cat_id, category in CATEGORIES.items():
        # Check category keywords
        cat_score = sum(1 for keyword in category.keywords if keyword in text_lower)
        
        # Check subcategory keywords
        for subcat in category.subcategories:
            subcat_score = sum(1 for keyword in subcat.keywords if keyword in text_lower)
            total_score = cat_score + subcat_score * 2  # Weight subcategory matches higher
            
            if total_score > highest_score:
                highest_score = total_score
                best_match = (cat_id, subcat.id)
    
    return best_match if highest_score > 0 else None

def get_all_categories() -> List[Dict[str, Any]]:
    """Get all categories as a list of dictionaries"""
    return [
        {
            "id": cat_id,
            "name": category.name,
            "description": category.description,
            "icon": category.icon,
            "color": category.color,
            "subcategories": [
                {
                    "id": subcat.id,
                    "name": subcat.name,
                    "description": subcat.description,
                    "icon": subcat.icon
                }
                for subcat in category.subcategories
            ]
        }
        for cat_id, category in CATEGORIES.items()
    ]

def get_category_hierarchy() -> Dict[str, List[str]]:
    """Get a simple hierarchy mapping for dropdowns"""
    return {
        category.name: [subcat.name for subcat in category.subcategories]
        for category in CATEGORIES.values()
    }