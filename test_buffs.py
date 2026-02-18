# Test Script for Wombat Spire Buff System

class Buff:
    def __init__(self, name, attack_mod=0, defense_mod=0):
        self.name = name
        self.attack_mod = attack_mod
        self.defense_mod = defense_mod

class Entity:
    def __init__(self, name, base_attack=10, base_defense=5):
        self.name = name
        self.base_attack = base_attack
        self.base_defense = base_defense
        self.buffs = []

    def add_buff(self, buff):
        self.buffs.append(buff)
        print(f"[{self.name}] Applied Buff: {buff.name}")

    def get_attack(self):
        mod = sum(b.attack_mod for b in self.buffs)
        return self.base_attack + mod

    def get_defense(self):
        mod = sum(b.defense_mod for b in self.buffs)
        return self.base_defense + mod

    def show_stats(self):
        print(f"[{self.name}] Current Stats - ATK: {self.get_attack()}, DEF: {self.get_defense()}")

def run_test():
    print("--- Starting Wombat Spire Buff System Test ---")
    
    player = Entity("Player")
    player.show_stats()
    
    # 1. Test Orthodox Buff
    orthodox_buff = Buff("Orthodox", attack_mod=5, defense_mod=5)
    player.add_buff(orthodox_buff)
    player.show_stats()
    
    # 2. Test Cult Buff
    cult_buff = Buff("Cult", attack_mod=30, defense_mod=-20)
    player.add_buff(cult_buff)
    player.show_stats()
    
    # 3. Multiple Buffs (Stacking)
    print("Adding another Orthodox Buff...")
    player.add_buff(orthodox_buff)
    player.show_stats()
    
    print("--- Test Finished ---")

if __name__ == "__main__":
    run_test()
