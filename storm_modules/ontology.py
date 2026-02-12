"""
STORM Ontology Module
Defines the hierarchical structure of research topics.
"""
from typing import List, Dict, Optional

class ResearchOntology:
    def __init__(self):
        # I. ÜST ALAN
        self.upper_field = "İletişim Bilimleri (Communication Studies)"
        
        # II. ANA ALAN
        self.main_field = "Dijital İletişim ve Medya Çalışmaları"
        
        # III. BİRİNCİL ALT ALANLAR (CORE DOMAINS)
        self.core_domains = {
            "Sosyal Medya ve Algoritmik İletişim": [
                "Algorithmic content curation",
                "Visibility regimes",
                "Engagement optimization",
                "Platform architecture",
                "Feed algorithms",
                "Content ranking mechanisms"
            ],
            "Otomatik Aktörler ve Yapay Etkileşim": [
                "Social bots",
                "Chatbots",
                "Automated content generation",
                "Synthetic users",
                "Bot-human interaction",
                "Botnets"
            ],
            "Aracılı Gerçeklik ve Simülasyon": [
                "Simulation society",
                "Mediated reality",
                "Crisis of representation",
                "Hyperreality",
                "Digital ontology"
            ]
        }
        
        # IV. İKİNCİL ALT ALANLAR (SUPPORTING DOMAINS)
        self.supporting_domains = {
            "Medya Ekolojisi": [
                "Evolution of media environments",
                "Digital environments",
                "Platforms as communication medium",
                "Information ecology"
            ],
            "Platform Çalışmaları (Platform Studies)": [
                "Platform capitalism",
                "Big tech corporations",
                "Content governance",
                "Moderation systems",
                "API economies"
            ],
            "Algoritmik Yönetişim ve Güç": [
                "Algorithmic power",
                "Digital censorship",
                "Politics of visibility",
                "Automated decision systems"
            ]
        }
        
        # V. YAN ALANLAR (CROSS-DISCIPLINARY)
        self.cross_disciplinary = {
            "Politik İletişim": [
                "Public opinion formation", 
                "Artificial agendas",
                "Algorithmic propaganda",
                "Astroturfing"
            ],
            "Enformasyon Bozukluğu Çalışmaları": [
                "Disinformation",
                "Misinformation",
                "Malinformation",
                "Information pollution"
            ],
            "İletişim Sosyolojisi": [
                "Digital alienation",
                "Collapse of participation",
                "Silent majority",
                "User passivity"
            ],
            "İnsan–Makine İletişimi": [
                "HCI / HMC",
                "Human-AI interaction",
                "Communicating with algorithms",
                "AI as social actor"
            ]
        }
        
        # VI. METODOLOJİK ALT ALANLAR
        self.methodologies = [
            "Computational social science",
            "Network analysis",
            "Automated content analysis",
            "Time series analysis",
            "Cross-platform analysis",
            "Web archival analysis"
        ]

    def get_all_topics(self) -> List[str]:
        """Returns a flat list of all sub-topics for broad search."""
        topics = []
        for domain in [self.core_domains, self.supporting_domains, self.cross_disciplinary]:
            for sublist in domain.values():
                topics.extend(sublist)
        return topics

    def get_domain_context(self, sub_topic: str) -> str:
        """Returns the parent domain for a given sub-topic to provide context."""
        for name, sublist in self.core_domains.items():
            if sub_topic in sublist: return f"Core Domain: {name}"
        for name, sublist in self.supporting_domains.items():
            if sub_topic in sublist: return f"Supporting Domain: {name}"
        for name, sublist in self.cross_disciplinary.items():
            if sub_topic in sublist: return f"Cross-Disciplinary: {name}"
        return "General Context"
