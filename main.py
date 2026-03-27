from sabrina_notebook_tdf import recommend_from_user_input as cosine_recommend
# from KNN import recommend_from_user_input as knn_recommend

def get_ingredients():
    '''
    Get a list of the ingredients the user would like to includ ein the recipe
    params: none
    return: list of ingredients
    '''

    ingredient_prompt = '''
    List any ingredients you would like to include (separated by comma).
    If none, hit Enter.
    ====================================================================
    '''
    ingredients = input(ingredient_prompt).split(',')
    # clean up the list to make sure it matches what will be in the dataset
    for i in range(len(ingredients)):
        ingredients[i] = ingredients[i].strip().lower()

    return ingredients

def get_dish_type(tags: list):
    '''
    Ask user about what kind of sish they want to make to get keyword tags
    params:
    returns:
    '''
    # ask about what kind of dish they are cooking
    dish_type_prompt = '''
    What kind of dish do you wish to make?
    ======================================
    1. Breakfast
    2. Lunch
    3. Brunch
    4. Dinner
    5. Side dish
    6. Appetizer
    7. Snack
    8. Dessert
    9. Beverage
    0: Any
    ======================================
    '''
    dish_type_keywords = {1: ['breakfast', 'brunch'], 2: ['lunch', 'brunch'], 3: ['breakfast', 'lunch', 'brunch'], 4: ['dinner', 'dinner-party', 'main-dish'],
                          5: ['appetizers', 'snacks', 'side-dishes'], 6: ['appetizers', 'snacks', 'side-dishes'], 7: ['appetizers', 'snacks', 'side-dishes'],
                          8: ['dessert'], 9: [], 0: []}
    
    valid_dish = False
    while not valid_dish:
        try:
            dish_type = int(input(dish_type_prompt))
            for keyword in dish_type_keywords[dish_type]:
                # could also use .extend here if time allows to fix all instances
                tags.append(keyword)
            valid_dish = True
        except (KeyError, ValueError):
            print('Invalid dish type. Try again.')

    #specify what kind of bev if selected bev
    bev_prompt = '''
    What kind of beverage?
    ======================
    1. Alcoholic
    2. Non-alchoholic
    3. Smoothie
    ======================
    '''
    bev_type_keywords = {1: ['beverages', 'cocktail'], 2: ['beverages', 'non-alchoholic'], 3: ['beverage', 'smoothie']}
    if dish_type == 9:
        valid_drink = False
        while not valid_drink:
            try: 
                bev_type = int(input(bev_prompt))
                for keyword in bev_type_keywords[bev_type]:
                    tags.append(keyword)
                valid_drink = True
            except (KeyError, ValueError):
                print('Invalid beverage type. Try again.')

    return tags

def get_dietary_restrictions(tags: list, numerics: dict):
    '''
    Inquire about dietary restrictions and update the tags with the relevent keywords
    and update the numerics based on what the user wants

    params: 
    - tags: list of keyword tags
    - numerics: dictionary of numerical search data

    returns:
    - tags updated with all relevent keywords
    - numerics updated with all relevent numerical info
    '''

    diet_rest_prompt = '''
    Do you have any dietary restrictions? Enter all restrictions (separated by comma).
    ==================================================================================
    1. General health diet
    2. Specific health diet
    3. Vegetarian
    4. Vegan
    5. Halal
    6. Kosher
    7. Gluten free
    8. Dairy free
    9. Nut free
    0: None
    ==================================================================================
    '''

    health_prompt = '''
    What are your diet specifications? Enter all specifications (separated by comma).
    =================================================================================
    1. Low carb
    2. Low fat
    3. Low sodium
    4. Low calorie
    5. Low cholesteral
    6. Low sugar
    7. Diabetic
    8. Other
    =================================================================================
    '''

    other_prompt = '''
    Please specify your dietary restriction (separated by comma if multiple).
    =========================================================================
    '''

    diet_rest = {1: ['healthy', 'healthy-2', 'low-in-something', 'diet'], 2: [], 3: ['vegetarian', 'vegan'], 4: ['vegan'],
                 5: ['halal', 'vegetarian', 'vegan', 'non-alchoholic'], 6: ['kosher'], 7: ['gluten-free'], 8: ['dairy-free', 'vegan'], 9: ['nut-free'], 0:[]}
    
    health_rest = {1: ['low-carb', 'very-low-carb'], 2: ['low-fat', 'low-saturated-fat'], 3: ['low-sodium'], 4: ['low-calorie'], 5: ['low-cholesteral'], 6: ['low-sugar'], 7: ['diabetic'], 8: []}

    valid_diet = False
    while not valid_diet:
        try:
            diet = input(diet_rest_prompt).split(',')
            for d in diet:
                d = int(d.strip())
                for keyword in diet_rest[d]:
                    tags.append(keyword)
                # update numerics with the specs
                if d == 1:
                    numerics['calories'] = 200
                    numerics['total_fat'] = 10
                    numerics['sugar'] = 10
                    numerics['sodium'] = 200
                    numerics['saturated_fat'] = 5
                    numerics['carbs'] = 20
                # if user wants to specify their dietary contraints
                if d == 2:
                    valid_spec = False
                    while not valid_spec:
                        try:
                            health_spec = input(health_prompt).split(',')
                            for h in health_spec:
                                h = int(h.strip())
                                for keyword in health_rest[h]:
                                    tags.append(keyword)
                                    for kw in diet_rest[1]:
                                        tags.append(kw)
                                valid_spec = True
                                if h == 1:
                                    numerics['carbs'] = 15
                                elif h == 2:
                                    numerics['total_fat'] = 7
                                    numerics['saturated_fat'] = 3
                                elif h == 3:
                                    numerics['sodium'] = 150
                                elif h == 4:
                                    numerics['calories'] = 200
                                elif h == 5:
                                    numerics['cholesteral'] = 100
                                elif h == 6:
                                    numerics['sugar'] = 10
                                elif h == 7:
                                    numerics['sugar'] = 10
                                    numerics['carbs'] = 15
                                elif h == 8:
                                    other_spec = input(other_prompt).split(',')
                                    for os in other_spec:
                                        tags.append(os.replace(' ', '-').strip().lower())
                        except (KeyError, ValueError):
                            print('Invalid diet specification. Try again.')
            valid_diet = True
        except (KeyError, ValueError):
            print('Invalid dietary restriction. Try again')

    return tags, numerics

def get_crowd_info(tags: list):
    '''
    Ask user if they are cooking for a crowd or not
    Add keywords to tags based on number of people they are cooking for

    params: 
    - tags: list of keywords

    returns:
    - tags: updated list of keywords
    '''
    crowd_prompt = '''
    Are you cooking for a crowd (y/n)?
    ==================================
    '''
    crowd = input(crowd_prompt).lower()
    if crowd == 'y':
        group_keywords = ['potluck', 'party-event', 'for-large-groups', 'party']
        for gkw in group_keywords:
            tags.append(gkw)
    elif crowd == 'n':
        tags.append('for-1-or-2')

    return tags
    
def get_easy(tags: list, numerics: dict):
    '''
    Ask the user if they need an easy recipe
    If yes, update the numerics and tags to reflect that
    Else, make no updates
    '''
    ease_prompt = '''
    Do you need an easy recipe (y/n)?
    =================================
    '''
    easy = input(ease_prompt).lower()
    if easy == 'y':
        easy_keywords = ['easy', 'beginner-cook', 'beginner']
        for ekw in easy_keywords:
            tags.append(ekw)
        numerics['n_steps'] = 6

    return tags, numerics

def get_budget(tags: list, numerics: dict):
    '''
    Ask the user if they are on a budget for a cheap recipe
    If yes, updated the tags and numerics to reflect that
    Else, make no change

    params:
    - tags: list of keywords
    - numerics: dict of numerical values

    returns:
    - tags: updated list of keywords
    - numerics: dict of updated numerical values
    '''
    
    budget_prompt = '''
    Are we ballin on a budget (y/n)?
    ================================
    '''
    budget = input(budget_prompt).lower()
    if budget == 'y':
        tags.extend(['inexpensive', 'budget'])
        numerics['n_ingredients'] = 5

    return tags, numerics

def get_time_cotraints(tags: list, numerics: dict):
    '''
    Ask the user about any time contraints they have
    Update the tags with the tags if they need a short cook time
    Update the numerics with the input amount of max time

    params: 
    - tags: list of keywords
    - numerics: dict of numerical values

    returns:
    - tags: updated list of keywrods if they require a quick recipe
    - numerics: updated dictionary with amoutn of time
    '''

    minutes_prompt = '''
    What is the maximum amount of time you have to cook (in minutes)?
    =================================================================
    '''

    valid_mins = False
    while not valid_mins:
        try:
            minutes = int(input(minutes_prompt).lower().replace('min', '').replace('m', '').strip())
            numerics['minutes'] = minutes
            if minutes < 15:
                tags.append('15-minutes-or-less')
                tags.append('quick')
            elif 15 < minutes <=30:
                tags.append('30-minutes-or-less')
                tags.append('quick')
            elif 30 < minutes <= 60:
                tags.append('60-minutes-or-less')
            valid_mins = True
        except (ValueError):
            print('Invalid time input. Try again.')

    return tags, numerics

def get_top_k():

    top_k_prompt = '''
    How many recipes would you like to choose from?
    ===============================================
    '''
    valid_top_k = False
    while not valid_top_k:
        try:
            top_k = int(input(top_k_prompt))
            valid_top_k = True
        except ValueError:
            print('Invalid response. Enter a number. Try again.')
    
    return top_k    

def option_1(tags: list, numerics: dict):
    '''
    Get any keywords relevent to the user preferences

    params: 
    - tags: list of keyword tags
    - numerics: dictionary of numerical search data

    returns:
    - tags updated with all relevent keywords
    - numerics updated with all relevent numerical info
    '''

    tags = get_dish_type(tags=tags)

    ingredients = get_ingredients()

    tags, numerics = get_dietary_restrictions(tags=tags, numerics=numerics)
    
    tags = get_crowd_info(tags=tags)

    tags, numerics = get_easy(tags=tags, numerics=numerics)

    tags, numerics = get_budget(tags=tags, numerics=numerics)

    tags, numerics = get_time_cotraints(tags=tags, numerics=numerics)

    top_k = get_top_k()
    
    return ingredients, tags, numerics, top_k

def get_keywrods(tags: list):

    keywords_prompt = '''
    Enter the keywords you would like to search by (separated by comma).
    ====================================================================
    '''

    keywords = input(keywords_prompt).split(',')
    for kw in keywords:
        kw = kw.replace(' ', '-').strip().lower()
        tags.append(kw)

    return tags

def option_2(tags: list, numerics: dict):
    '''
    Serch by specific things
    '''

    # TODO teh random option
    explore_options = '''
    Search by (separate options by comma if more than 1):
    =====================================================
    1. Ingredients
    2. Keywords
    3. Type of dish
    4. Time
    5. Dietary Restriction
    6. Budget
    7. Serving size
    8. Easy
    9. Random    
    x. Back
    =====================================================
    '''

    ingredients = []
    valid_exp_option = False
    while not valid_exp_option:
        try:
            exp_option = input(explore_options).split(',')
            if 'x' in exp_option or 'X' in exp_option:
                return [], tags, numerics, 3, True
            for opt in exp_option:
                opt = int(opt)
                if opt == 1:
                    ingredients = get_ingredients()
                elif opt == 2:
                    tags = get_keywrods(tags=tags)
                elif opt == 3:
                    tags = get_dish_type(tags=tags)
                elif opt == 4:
                    tags, numerics = get_time_cotraints(tags=tags, numerics=numerics)
                elif opt == 5:
                    tags, numerics = get_dietary_restrictions(tags=tags, numerics=numerics)
                elif opt == 6:
                    tags, numerics = get_budget(tags=tags, numerics=numerics)
                elif opt == 7:
                    tags = get_crowd_info(tags=tags)
                elif opt == 8:
                    tags, numerics = get_easy(tags=tags, numerics=numerics)
                elif opt == 9:
                    # TODO !!
                    print('Random option coming soon')
            valid_exp_option = True    

        except ValueError:
            print('Invalid exploration option. Try again')
    
    top_k = get_top_k()

    return ingredients, tags, numerics, top_k, False


def get_menu_option():

    menu = '''
    Recipe Recommendation System
    Enter the number of your selection.
    ================================================================
    1. Get recommendations based on your input
    2. Explore recipes
    3. Exit
    ================================================================
    '''
    valid_option = False
    while not valid_option:
        try: 
            option = int(input(menu))
            valid_option = True
        except ValueError:
            print('Invalid menu option. Try again.')

    return option


def main():

    # our main menu that will run this program basically
    # TODO do we want to add a "would you like to go again" option to the program?
    
    option = get_menu_option()

    while option != 3:

        user_tags = []
        user_numerics = {'minutes': 60, 'n_steps': 10, 'n_ingredients': 15} 
            # 'calories': 1000, 'total_fat': 50, 'sugar': 50, 'sodium': 1000,
            # 'protein': 30, 'saturated_fat': 5, 'carbs': 50}

        # first option: get recipe recs
        if option == 1:

            
            user_ingredients, user_tags, user_numerics, top_k = option_1(tags=user_tags, numerics=user_numerics)
            print('Ingredients:', user_ingredients)
            print('Tags:', user_tags)
            print('Numerics:', user_numerics)
            print('k:', top_k)
            results = cosine_recommend(
                user_ingredients = user_ingredients,
                user_tags        = user_tags,
                user_numeric     = user_numerics,
                top_k            = top_k
            )
            print(results.to_string())
            # recommend_from_user_input(
            # user_ingredients=["chicken", "garlic", "olive oil"],
            # user_tags=["dinner", "quick"],
            # user_numeric={"minutes": 30, "n_steps": 6, "n_ingredients": 8},
            # user_name="lemon garlic chicken",
            # top_k=5
            # )

            # print(results)
            

        elif option == 2:
            user_ingredients, user_tags, user_numerics, top_k, back = option_2(tags=user_tags, numerics=user_numerics)
            if not back:
                print('Ingredients:', user_ingredients)
                print('Tags:', user_tags)
                print('Numerics:', user_numerics)
                print('k:', top_k)
                            # results = recommend_from_user_input(
            #     user_ingredients = user_ingredients,
            #     user_tags        = user_tags,
            #     user_numeric     = user_numeric,
            #     top_k            = top_k
            # )

            # recommend_from_user_input(
            # user_ingredients=["chicken", "garlic", "olive oil"],
            # user_tags=["dinner", "quick"],
            # user_numeric={"minutes": 30, "n_steps": 6, "n_ingredients": 8},
            # user_name="lemon garlic chicken",
            # top_k=5
            # )

            # print(results)

        else:
            print('Invalid menu option, try again')

        option = get_menu_option()

if __name__ == "__main__":
    main()