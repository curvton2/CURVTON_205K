
# VTON Matching Rules
# Dictionary defining compatible and incompatible pairings

# Category definitions
CATEGORIES = {
    "upper": ["T-shirt", "Polo", "Shirt", "Tank-top", "Blouse", "Sweater", "Hoodie", "Jacket", "Windbreaker", "Coat", "Blazer", "Kurta", "Hanfu-style top", "Dashiki", "Gym-top", "Lab-coat"],
    "lower": ["Jeans", "Trousers", "Chinos", "Shorts", "Skirt", "Dhoti", "Lungi"],
    "full": ["Suit (two-piece)", "Dress (casual)", "Dress (formal)", "Maxi-dress", "Sari", "Salwar-Kameez", "Sherwani", "Deel (traditional Mongol robe)", "Kimono", "Yukata", "Kaftan", "Abaya", "Kente-wrapper", "Poncho", "Parkas", "Puffer-jacket", "Athletic-wear", "School-uniform", "Work-uniform", "Traditional robe", "Traditional dress", "Swimwear", "Nightwear (pajamas)"]
}

def get_category(garment_name):
    for cat, items in CATEGORIES.items():
        if garment_name in items:
            return cat
    return "unknown"

def is_compatible(currently_worn_list, target_garment):
    """
    Checks if the target_garment can be naturally paired with what the person is wearing.
    currently_worn_list: List of garment names the person has on (from edit.jsonl).
    target_garment: The category name of the cloth being tried on.
    """
    target_cat = get_category(target_garment)
    
    # Conflict rules:
    # 1. If target is Full-Body, it's generally compatible (replaces everything)
    if target_cat == "full":
        return True
        
    # 2. If target is Upper/Lower, check for "Full-Body" conflicts on the person
    # Example: Don't put a T-shirt OVER a Sari or a formal Dress.
    worn_categories = [get_category(g) for g in currently_worn_list]
    
    if "full" in worn_categories:
        # Exceptions: Jackets/Coats can go over some full body items
        if target_garment in ["Jacket", "Coat", "Windbreaker", "Blazer", "Poncho"]:
            return True
        return False # No T-shirts over Dresses
        
    # 3. Simple Upper/Lower logic:
    # If target is Upper and person already has an Upper, it's a replacement (Compatible)
    # If target is Lower and person already has a Lower, it's a replacement (Compatible)
    
    return True

MALE_CLOTH_MATCHES = {
    "all": CATEGORIES["upper"] + CATEGORIES["lower"] + CATEGORIES["full"]
}

FEMALE_CLOTH_MATCHES = {
    "all": CATEGORIES["upper"] + CATEGORIES["lower"] + CATEGORIES["full"]
}

def get_matching_rule(gender, cloth_id):
    if gender == "male":
        return MALE_CLOTH_MATCHES.get(cloth_id, MALE_CLOTH_MATCHES["all"])
    else:
        return FEMALE_CLOTH_MATCHES.get(cloth_id, FEMALE_CLOTH_MATCHES["all"])
