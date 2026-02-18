import pygame
import sys

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)

class Buff:
    def __init__(self, name, attack_mod=0, defense_mod=0, color=WHITE):
        self.name = name
        self.attack_mod = attack_mod
        self.defense_mod = defense_mod
        self.color = color

class Entity:
    def __init__(self, name, x, y, color):
        self.name = name
        self.max_hp = 100
        self.hp = 100
        self.base_attack = 10
        self.base_defense = 5
        self.buffs = []
        self.rect = pygame.Rect(x, y, 100, 100)
        self.color = color

    def add_buff(self, buff):
        self.buffs.append(buff)

    def get_attack(self):
        mod = sum(b.attack_mod for b in self.buffs)
        return self.base_attack + mod

    def get_defense(self):
        mod = sum(b.defense_mod for b in self.buffs)
        return self.base_defense + mod

    def draw(self, screen, font):
        # Draw body
        pygame.draw.rect(screen, self.color, self.rect)
        
        # Draw Health Bar
        bar_width = 100
        bar_height = 10
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y + 110, bar_width, bar_height))
        current_bar_width = int(bar_width * (self.hp / self.max_hp))
        pygame.draw.rect(screen, GREEN, (self.rect.x, self.rect.y + 110, current_bar_width, bar_height))

        # Draw Text (Name, Stats)
        name_surf = font.render(self.name, True, BLACK)
        screen.blit(name_surf, (self.rect.x, self.rect.y - 30))
        
        stats_text = f"ATK:{self.get_attack()} DEF:{self.get_defense()}"
        stats_surf = font.render(stats_text, True, BLACK)
        screen.blit(stats_surf, (self.rect.x, self.rect.y + 130))

        # Draw Buffs
        for i, buff in enumerate(self.buffs):
            pygame.draw.circle(screen, buff.color, (self.rect.x + 10 + i*20, self.rect.y + 160), 8)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Wombat Spire - Combat Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    player = Entity("Player (Wombat)", 150, 250, BLUE)
    enemy = Entity("Enemy (Monster)", 550, 250, RED)

    # Predefined Buffs
    orthodox_buff = Buff("Orthodox", attack_mod=5, defense_mod=5, color=GOLD)
    cult_buff = Buff("Cult", attack_mod=30, defense_mod=-20, color=PURPLE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    player.add_buff(orthodox_buff)
                    print(f"Applied Orthodox Buff to Player. New ATK: {player.get_attack()}, DEF: {player.get_defense()}")
                if event.key == pygame.K_2:
                    player.add_buff(cult_buff)
                    print(f"Applied Cult Buff to Player. New ATK: {player.get_attack()}, DEF: {player.get_defense()}")
                if event.key == pygame.K_r:
                    player.buffs = []
                    print("Reset Player Buffs.")

        screen.fill(WHITE)
        
        # UI Instructions
        instr_1 = font.render("Press '1' for Orthodox Buff (+5 ATK, +5 DEF)", True, BLACK)
        instr_2 = font.render("Press '2' for Cult Buff (+30 ATK, -20 DEF)", True, BLACK)
        instr_r = font.render("Press 'R' to Reset Buffs", True, BLACK)
        screen.blit(instr_1, (20, 20))
        screen.blit(instr_2, (20, 50))
        screen.blit(instr_r, (20, 80))

        player.draw(screen, font)
        enemy.draw(screen, font)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
