# manim -pqh visualizations/rlvr_animation.py RLVRDecomposition
from manim import *

class RLVRDecomposition(Scene):
    def construct(self):
        # Set white background
        self.camera.background_color = WHITE
        
        # Original equation
        original = MathTex(
            r"\mathcal{L}_{\mathrm{RLVR}}(\theta)", 
            r"=", 
            r"-\mathbb{E}\left[\sum_{y} r(x, y)\cdot\pi_{\theta}(y|x)\right]",
            r",",
            r"\quad r(x,y) \in \{-1, +1\}",
            color=BLACK,
            font_size=44
        )
        original.to_edge(UP, buff=1.0)
        self.add(original)
        self.wait(1.2)
        
        # Second line - equals sign
        equals_sign = MathTex(
            r"=",
            font_size=44,
            color=BLACK
        )
        
        # PSR part - using \mathbb{E} instead of full expectation
        psr_part = MathTex(
            r"-\mathbb{E}\left[\sum_{y:r(x,y)=1}\pi_\theta(y|x)\right]",
            font_size=44,
            color=GREEN
        )
    
        # NSR part
        nsr_part = MathTex(
            r"-\mathbb{E}\left[\sum_{y:r(x,y)=-1}-\pi_\theta(y|x)\right]",
            font_size=44,
            color=RED
        )
        
        # Position second line
        equals_sign.next_to(original, DOWN, buff=1.5)
        equals_sign.align_to(original[1], LEFT)
        
        psr_part.next_to(equals_sign, RIGHT, buff=0.15)
        nsr_part.next_to(psr_part, RIGHT, buff=0.1)
        
        second_line = VGroup(equals_sign, psr_part, nsr_part)
        
        self.play(Write(second_line))
        self.wait(1.5)
        
        # Add underbraces
        psr_brace = Brace(psr_part, direction=DOWN, color=GREEN)
        psr_label = MathTex(r"\mathcal{L}_{\mathrm{PSR}}(\theta)", color=GREEN, font_size=32)
        psr_label.next_to(psr_brace, DOWN, buff=0.1)
        
        nsr_brace = Brace(nsr_part, direction=DOWN, color=RED)
        nsr_label = MathTex(r"\mathcal{L}_{\mathrm{NSR}}(\theta)", color=RED, font_size=32)
        nsr_label.next_to(nsr_brace, DOWN, buff=0.1)
        
        self.play(
            GrowFromCenter(psr_brace),
            Write(psr_label),
            GrowFromCenter(nsr_brace),
            Write(nsr_label)
        )
        self.wait(2.5)