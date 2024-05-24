import time

import matplotlib.pyplot as plt
from csv import DictReader
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

import requests
from PIL import Image
from matplotlib.animation import PillowWriter
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
from collections import defaultdict
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

warnings.filterwarnings("ignore")


class Data:
    def __init__(self):
        self.aeroflot_podatki = []
        adtype = [
            ('Date', 'U10'),  # Date as string
            ('Location', 'U100'),  # Location as string
            ('Aircraft', 'U100'),  # Aircraft as string
            ('Tail number', 'U100'),  # Tail number as string
            ('Airline division', 'U100'),  # Airline division as string
            ('Aircraft damage', 'U100'),  # Aircraft damage as string,
            ('Fatalities', 'U100'),  # Fatalities as string,
            ('Description', 'U1000'),  # Description as string,
            ('Refs', 'U100')  # Refs as string
        ]
        self.podatki = []
        dtype = [
            ('Date', 'U10'),  # Date as string
            ('Time', 'U10'),  # Time as string
            ('Location', 'U100'),  # Location as string
            ('Operator', 'U100'),  # Operator as string
            ('Flight', 'U100'),  # Flight number as string
            ('Route', 'U100'),  # Route as string
            ('Type', 'U100'),  # Aircraft type as string
            ('Registration', 'U100'),  # Registration as string
            ('cn_In', 'U100'),  # Construction number or other identifier as string
            ('Aboard', np.int32),  # Number of people aboard as integer
            ('Fatalities', np.int32),  # Number of fatalities as integer
            ('Ground', np.int32),  # Number of ground fatalities as integer
            ('Summary', 'U1000')  # Summary as string
        ]

        with open("podatki\\Airplane_Crashes_and_Fatalities_Since_1908.csv", 'r', encoding='utf-8') as file:
            csv_reader = DictReader(file)
            for row in csv_reader:
                date = row['Date']
                time = row['Time']
                location = row['Location']
                operator = row['Operator']
                flightNr = row['Flight #']
                route = row['Route']
                type = row['Type']
                registration = row['Registration']
                CnIn = row['cn/In']
                aboard = int(row["Aboard"]) if row["Aboard"] else 0
                fatalities = int(row["Fatalities"]) if row["Fatalities"] else 0
                ground = int(row['Ground']) if row["Ground"] else 0
                summary = row['Summary']
                self.podatki.append(
                    (date, time, location, operator, flightNr, route, type, registration, CnIn, aboard, fatalities,
                     ground, summary))
        self.podatki = np.sort(np.array(self.podatki, dtype=dtype), order='Date')

        with open("podatki\\aeroflot_accidents_1970s.csv", 'r', encoding='utf-8') as file:
            csv_reader = DictReader(file)
            for row in csv_reader:
                date = row['Date']
                location = row['Location']
                aircraft = row['Aircraft']
                tail_number = row['Tail number']
                airline_division = row['Airline division']
                aircraft_damage = row['Aircraft damage']
                fatalities = row['Fatalities']
                description = row['Description']
                refs = row['Refs']
                self.aeroflot_podatki.append(
                    (date, location, aircraft, tail_number, airline_division, aircraft_damage, fatalities, description,
                     refs)
                )
        self.aeroflot_podatki = np.sort(np.array(self.aeroflot_podatki, dtype=adtype), order='Date')


        self.years = np.unique([entry['Date'][-4:] for entry in self.podatki])
        self.years_numeric = np.array([int(year) for year in self.years])
        self.num_accidents = np.array(
            [np.sum([1 for entry in self.podatki if entry['Date'].endswith(year)]) for year in self.years],
            dtype=np.float64)
        self.num_fatalities = np.array(
            [np.sum([entry['Fatalities'] for entry in self.podatki if entry['Date'].endswith(year)]) for year in
             self.years], dtype=np.float64)
        self.num_passengers = np.array(
            [np.sum([entry['Aboard'] for entry in self.podatki if entry['Date'].endswith(year)]) for year in
             self.years], dtype=np.float64)
        self.ratio = self.num_accidents / self.num_passengers * 100

    def get_geolocations_old(self):
        # ZEMLJEVID SVETA - Pridobivanje koordinat in shranjevanje v .csv datoteko
        # !! PROGRAM TEČE OKOLI 3 URE !! PODATKI SO ŽE V CSV DATOTEKI !!
        base_url = "https://nominatim.openstreetmap.org/search.php"
        koordinate = []
        a = 0
        for entry in self.podatki:
            time.sleep(2)
            location = re.sub(
                r'^(?:' + '|'.join(re.escape(prefix) for prefix in ["Near", "Off", "Around", "Close to"]) + r')\s*',
                '',
                entry['Location'])
            params = {
                'q': location,
                'format': 'json'
            }
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    print("Display Name:", data[0]['display_name'])
                    print("Latitude:", data[0]['lat'])
                    print("Longitude:", data[0]['lon'])
                    data[0]['date'] = entry['Date']
                    koordinate.append(data[0])
                    # print(data[0])
                    if (a == 5):
                        break
                    a += 1
                else:
                    print("No results found")

            else:
                print("Failed to fetch data", location)

        crash_data = pd.DataFrame({
            'Location Name': [koordinata['display_name'] for koordinata in koordinate],
            'Date': [koordinata['date'] for koordinata in koordinate],
            'Latitude': [float(koordinata['lat']) for koordinata in koordinate],
            'Longitude': [float(koordinata['lon']) for koordinata in koordinate]
        })
        # Shrani DataFrame v .csv datoteko
        crash_data.to_csv('podatki/koordinate.csv', index=False)

    def get_geolocations(self):
        # ZEMLJEVID SVETA - Pridobivanje koordinat in shranjevanje v .csv datoteko
        # !! PROGRAM TEČE OKOLI 3 URE !! PODATKI SO ŽE V CSV DATOTEKI !!
        geolocator = Nominatim(user_agent="pr")
        koordinate = []
        for entry in self.podatki:
            time.sleep(2)
            while(True):
                location = re.sub(
                    r'^(?:' + '|'.join(re.escape(prefix) for prefix in ["Near", "Off", "Around", "Close to"]) + r')\s*',
                    '',
                    entry['Location'])
                try:
                    geocode_result = geolocator.geocode(location)
                    if geocode_result:
                        print("Display Name:", geocode_result.address)
                        print("Latitude:", geocode_result.latitude)
                        print("Longitude:", geocode_result.longitude)
                        koordinate.append({
                            'display_name': geocode_result.address,
                            'lat': geocode_result.latitude,
                            'lon': geocode_result.longitude,
                            'date': entry['Date']
                        })
                    else:
                        print("No results found for location:", location)
                    break
                except Exception as e:
                    print("Failed to fetch data for location:", location, "Error:", str(e))
                    time.sleep(60)

        crash_data = pd.DataFrame({
            'Location Name': [koordinata['display_name'] for koordinata in koordinate],
            'Date': [koordinata['date'] for koordinata in koordinate],
            'Latitude': [koordinata['lat'] for koordinata in koordinate],
            'Longitude': [koordinata['lon'] for koordinata in koordinate]
        })
        # Shrani DataFrame v .csv datoteko
        crash_data.to_csv('podatki/koordinate.csv', index=False)

    def crashes_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.num_accidents, marker='o', linestyle='-')
        plt.title('Number of Airplane Accidents Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Accidents')
        plt.grid(True)
        plt.xticks(self.years[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def crashes_over_time_regression(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.years_numeric, self.num_accidents, marker='o')

        p = np.poly1d(np.polyfit(self.years_numeric, self.num_accidents, 3))
        plt.plot(self.years_numeric, p(self.years_numeric))
        plt.title('Number of Airplane Accidents Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Accidents')
        plt.grid(True)
        plt.xticks(self.years_numeric[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def ratio_over_time(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.ratio, marker='o', linestyle='-')
        plt.title('Number of accidents / Total amount of passegers Ratio by Year')
        plt.xlabel('Year')
        plt.ylabel('Ratio (%)')
        plt.grid(True)
        plt.xticks(self.years[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def ratio_over_time_regression(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.years_numeric, self.ratio, marker='o')

        p = np.poly1d(np.polyfit(self.years_numeric, self.ratio, 3))
        plt.plot(self.years_numeric, p(self.years_numeric))
        plt.title('Number of accidents / Total amount of passegers Ratio by Year')
        plt.xlabel('Year')
        plt.ylabel('Ratio (%)')
        plt.grid(True)
        plt.xticks(self.years_numeric[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def fatilities_vs_ratio(self):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.subplots()
        ax1.plot(self.years, self.ratio, color='orange', marker=".", linewidth=1)
        ax1.set_xlabel('Year', fontsize=11)
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax1.set_ylabel('Ratio (%)', color='orange', fontsize=11)
        ax1.tick_params('y', colors='orange')
        ax2 = ax1.twinx()
        ax2.plot(self.years, self.num_accidents, color='red', marker=".", linewidth=1)
        ax2.set_ylabel('Number of fatalities', color='red', fontsize=11)
        ax2.tick_params('y', colors='r')
        plt.title('Accidents VS Ratio by Year', loc='Center', fontsize=14)

        ax1.set_xticks(self.years[::5])
        for label in ax1.xaxis.get_ticklabels()[::5]:
            label.set_visible(True)
        for label in ax1.xaxis.get_ticklabels()[1::5]:
            label.set_visible(False)

        fig.tight_layout()
        plt.show()

    def deaths_over_time(self):
        years = np.array([int(entry[0][-4:]) for entry in self.podatki])  # Extract years from the date
        fatalities = np.array([entry[10] for entry in self.podatki])  # Extract fatalities data

        plt.figure(figsize=(18, 8))
        plt.plot(fatalities, years, 'o')
        plt.xlabel('Fatalities')
        plt.ylabel('Years')
        plt.show()

    def ratio_between_aboard_fatal(self):
        aboard = np.array([entry[9] for entry in self.podatki])
        fatalities = np.array([entry[10] for entry in self.podatki])

        survivors = aboard - fatalities

        plt.figure(figsize=(8, 8))
        labels = ['Survivors', 'Fatalities']
        sizes = [np.sum(survivors), np.sum(fatalities)]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Survivors to Fatalities')
        plt.axis('equal')
        plt.show()

    def fatilities_vs_ratio_regression(self):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.subplots()

        plt.scatter(self.years_numeric, self.ratio, marker='o', color='orange')
        p = np.poly1d(np.polyfit(self.years_numeric, self.ratio, 3))
        plt.plot(self.years_numeric, p(self.years_numeric), color='orange')
        ax1.set_xlabel('Year', fontsize=11)
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax1.set_ylabel('Ratio (%)', color='orange', fontsize=11)
        ax1.tick_params('y', colors='orange')
        ax2 = ax1.twinx()

        plt.scatter(self.years_numeric, self.num_accidents, marker='o', color='red')
        p = np.poly1d(np.polyfit(self.years_numeric, self.num_accidents, 3))
        plt.plot(self.years_numeric, p(self.years_numeric), color='red')
        ax2.set_ylabel('Number of fatalities', color='red', fontsize=11)
        ax2.tick_params('y', colors='r')
        plt.title('Accidents VS Ratio by Year', loc='Center', fontsize=14)

        ax1.set_xticks(self.years_numeric[::5])
        for label in ax1.xaxis.get_ticklabels()[::5]:
            label.set_visible(True)
        for label in ax1.xaxis.get_ticklabels()[1::5]:
            label.set_visible(False)

        fig.tight_layout()
        plt.show()

    def highest_crash_counts(self):
        registrations = np.array([crash[6] for crash in self.podatki])

        unique_registrations, counts = np.unique(registrations, return_counts=True)

        sorted_indices = np.argsort(counts)[::-1]
        top_10_indices = sorted_indices[:10]
        top_10_airplanes = unique_registrations[top_10_indices]

        top_10_dict = {}

        for airplane in top_10_airplanes:
            count = counts[np.where(unique_registrations == airplane)[0][0]]
            print(airplane, ":", count)
            top_10_dict[airplane] = count

        airplanes = list(top_10_dict.keys())
        crash_counts = list(top_10_dict.values())

        plt.figure(figsize=(10, 6))
        plt.bar(airplanes, crash_counts, color='skyblue')
        plt.title('Top 10 Airplanes with Highest Crash Counts')
        plt.xlabel('Airplane Registration Number')
        plt.ylabel('Number of Crashes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def extract_data_from_summary(self, include, exclude, column):
        extracted = np.array([entry for entry in self.podatki if
                              any(inc in entry[column].lower() for inc in include)
                              and not
                              any(exc in entry[column].lower() for exc in exclude)])
        return extracted

    def operator_performance(self):
        # Izračunamo izraz (število žrtev * število potnikov) - število letal za vsako letalsko podjetje
        podjetja = np.unique([entry['Operator'] for entry in self.podatki])
        uspešnost = []
        for podjetje in podjetja:
            nesreče_podjetja = [entry for entry in self.podatki if entry['Operator'] == podjetje]
            število_letov = np.sum([1 for entry in self.podatki if entry['Operator'] == podjetje])
            if (število_letov > 5):
                skupno_mrtvih = np.sum([entry['Fatalities'] for entry in nesreče_podjetja])
                skupno_potnikov = np.sum([entry['Aboard'] for entry in nesreče_podjetja])

                izraz = (skupno_mrtvih * skupno_potnikov) / število_letov
                uspešnost.append(izraz)

        max_uspešnost = max(uspešnost)
        uspešnost = [x / max_uspešnost for x in uspešnost]

        # Pridobimo top 25 in lowest 25 letalskih podjetij glede na izraz
        sorted_indices_top = np.argsort(uspešnost)[::-1][:25]
        sorted_indices_lowest = np.argsort(uspešnost)[:25]
        top_podjetja = podjetja[sorted_indices_top]
        lowest_podjetja = podjetja[sorted_indices_lowest]
        top_uspešnost = np.array(uspešnost)[sorted_indices_top]
        lowest_uspešnost = np.array(uspešnost)[sorted_indices_lowest]

        # Priprava stolpičnih grafov
        plt.figure(figsize=(12, 8))

        # Top 25 letalskih podjetij
        plt.subplot(2, 1, 1)
        plt.bar(top_podjetja, top_uspešnost, color='skyblue')
        plt.xlabel('Letalsko podjetje')
        plt.ylabel('Uspešnost')
        plt.title('Najboljših 25 letalskih podjetij glede na uspešnost')
        plt.xticks(rotation=90)

        # Lowest 25 letalskih podjetij
        plt.subplot(2, 1, 2)
        plt.bar(lowest_podjetja, lowest_uspešnost, color='salmon')
        plt.xlabel('Letalsko podjetje')
        plt.ylabel('Uspešnost')
        plt.title('Najslabših 25 letalskih podjetij glede na uspešnost')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

    def crash_locations_on_map(self):
        crash_data = pd.read_csv("podatki/koordinate.csv")
        crash_data['Date'] = pd.to_datetime(crash_data['Date'], format='%m/%d/%Y')
        crash_data['Year'] = crash_data['Date'].dt.year
        crash_data = crash_data.sort_values(by='Year')

        years = crash_data['Year'].unique()

        fig, (ax_map, ax_bar) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [5, 1]})

        # Set up the map plot
        ax_map = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax_map.coastlines()
        ax_map.add_feature(cfeature.BORDERS)
        ax_map.add_feature(cfeature.LAND)
        ax_map.add_feature(cfeature.OCEAN)

        l = ax_map.scatter([], [], color='red', marker='o', label='nesreče po svetu', s=1)
        ax_map.legend()
        ax_map.set_title('Letalske nesreče po svetu', loc='center', pad=20)

        # Set up the bar plot
        ax_bar.set_xlim(years[0], years[-1])
        ax_bar.set_ylim(0, 1)
        bar = ax_bar.barh(0.5, 0, left=years[0], height=0.5, color='paleturquoise')
        ax_bar.axis('off')

        # Add text annotation for the current year, fixed at a position
        year_text = fig.text(0.5, 0.1, '', ha='center', va='center', fontsize=12, transform=fig.transFigure)

        writer = PillowWriter(fps=8, metadata=dict(title='World plot'))

        with writer.saving(fig, "world_map.gif", 100):
            for year in years:
                current_data = crash_data[crash_data['Year'] <= year]
                l.set_offsets(current_data[['Longitude', 'Latitude']])

                # Update the bar
                bar[0].set_width(year - years[0])

                # Update the year text
                #year_text.set_text(str(year))
                year_text.set_text(str(years[0]) + " - " + str(year))

                writer.grab_frame()

            # Hold the last frame for a while
            l.set_offsets(crash_data[['Longitude', 'Latitude']])
            bar[0].set_width(years[-1] - years[0])
            year_text.set_text(str(years[0]) + " - " + str(years[-1]))
            for _ in range(50):
                writer.grab_frame()

    def passanger_and_fatalities_over_time(self):

        plt.figure(figsize=(10, 6))
        plt.bar(self.years, self.num_passengers, color='black')
        plt.bar(self.years, self.num_passengers - self.num_fatalities,
                color=[(np.random.random(), np.random.random(), np.random.random()) for _ in range(len(self.years))])
        plt.title('Passangers and Fatalities over time')
        plt.xlabel('Year')
        plt.ylabel('Number of Accidents')
        plt.xticks(self.years[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def wordcloud(self, field):
        text = str(self.podatki[field])
        plane_mask = np.array(Image.open('assets/airplane_mask.jpg'))

        stopwords = set(STOPWORDS)

        wc = WordCloud(background_color="white", max_words=4000, mask=plane_mask,
                       stopwords=stopwords)
        wc.generate(text)

        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(field, loc='Center', fontsize=14)
        plt.show()

    def countrycloud(self, field):

        countries = []
        for location in self.podatki[field]:
            country = location.split(",")[-1].strip()
            country = re.sub(r'[^a-zA-Z\s]', '', country)
            if country:
                countries.append(country)

        text = ' '.join(countries)
        plane_mask = np.array(Image.open('assets/airplane_mask.jpg'))

        stopwords = set(STOPWORDS)
        stopwords.add('nan')
        stopwords.add('Near')

        wc = WordCloud(background_color="white", max_words=2000, mask=plane_mask,
                       stopwords=stopwords)

        wc.generate(text)

        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title('Location of Accident', loc='Center', fontsize=14)
        plt.show()

    def top_operators(self, field, n=3):

        operator_counts = Counter(self.podatki[field])
        top_operators = operator_counts.most_common(n)

        return top_operators

    def count_of_accidents_by_year_operator(self):
        accidents_by_year = defaultdict(int)

        for row in self.podatki:
            date = row['Date']
            operator = row['Operator']

            if 'Aeroflot' in operator:
                year = date.split('/')[-1]
                accidents_by_year[year] += 1

        accidents_by_year = dict(sorted(accidents_by_year.items()))

        years = list(accidents_by_year.keys())
        counts = list(accidents_by_year.values())

        plt.figure(figsize=(10, 6))
        plt.plot(years, counts, marker='o', linestyle='-')
        plt.title('Count of Accidents by Year for Aeroflot Operator')
        plt.xlabel('Year')
        plt.ylabel('Accident Count')
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()

    def type_of_flights(self):

        military_counts = []
        passenger_counts = []

        for year in self.years:
            military_count = np.count_nonzero([1 for entry in self.podatki if
                                               entry['Date'].endswith(year) and 'MILITARY' in entry[
                                                   'Operator'].upper()])
            passenger_count = np.count_nonzero([1 for entry in self.podatki if
                                                entry['Date'].endswith(year) and 'MILITARY' not in entry[
                                                    'Operator'].upper()])
            military_counts.append(military_count)
            passenger_counts.append(passenger_count)

        np.array(military_counts), np.array(passenger_counts)
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.years_numeric, military_counts, label='Military', color='red', marker='o')
        plt.plot(self.years_numeric, passenger_counts, label='Commercial', color='blue', marker='o')
        plt.xlabel('Years')
        plt.ylabel('Counts')
        plt.title('Accident Counts (Military vs Commercial)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        labels = ['Military', 'Commercial']
        counts = [np.sum(military_counts), np.sum(passenger_counts)]
        colors = ['red', 'blue']
        plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Accident Distribution (Military vs Commercial)')

        plt.tight_layout()
        plt.show()

    """def cluster_crashes_by_reason(self):
                # Extract descriptions from the data
        descriptions = [row['Description'] for row in self.aeroflot_podatki]

        # Convert descriptions into TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(descriptions)

        # Apply PCA to reduce the dimensionality to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())

        # Apply K-means clustering
        num_clusters = 5  # You can adjust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # Assign cluster labels to each description
        cluster_labels = kmeans.labels_

        # Plot the clusters on a 2D scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', legend='full')
        plt.title('Clustering of Crashes by Reason')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

        # Group descriptions by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(descriptions[i])

        return clusters"""

    def cluster_crashes_by_reason(self):
        # Extract descriptions from the data
        descriptions = [row['Description'] for row in self.aeroflot_podatki]

        custom_stop_words = ['airport',
                             'aircraft',
                             'flight',
                             'crashed',
                             'flight',
                             'english',
                             'the',
                             'to',
                             'was',
                             'and',
                             'on',
                             'in',
                             'while',
                             'too',
                             'of',
                             'at',
                             'off',
                             'after',
                             'shortly',
                             'from',
                             'an',
                             'en',
                             'into',
                             'during',
                             'failure',
                             'route',
                             'taking',
                             'approach']

        # Convert descriptions into TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=custom_stop_words)
        X = vectorizer.fit_transform(descriptions)

        # Apply PCA to reduce the dimensionality to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())

        # Apply K-means clustering
        num_clusters = 3  # You can adjust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # Get cluster centroids
        cluster_centers = kmeans.cluster_centers_

        # Get top keywords for each cluster
        top_keywords = []
        for i, centroid in enumerate(cluster_centers):
            top_keyword_indices = centroid.argsort()[-5:][::-1]  # Get indices of top 5 keywords
            top_keywords.append([vectorizer.get_feature_names_out()[index] for index in top_keyword_indices])

        # Collect example descriptions for each cluster
        examples_per_cluster = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            examples_per_cluster[label].append(descriptions[i])

        # Plot the clusters on a 2D scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.labels_, palette='viridis', legend='full')
        plt.title('Clustering of Crashes by Reason')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Add legend with top keywords for each cluster
        for i, keywords in enumerate(top_keywords):
            keywords_str = ', '.join(keywords)
            plt.text(X_pca[kmeans.labels_ == i, 0].mean(), X_pca[kmeans.labels_ == i, 1].mean(),
                     f'Cluster {i}: {keywords_str}', color='black', fontsize=10)

        for label, examples in examples_per_cluster.items():
            print(f"Cluster {label}:")
            for example in examples[:5]:  # Print up to 5 examples per cluster
                print(example)
            print()

        plt.tight_layout()
        plt.show()

    def cluster_crashes_by_reason_vsi(self):
        # Extract descriptions from the data
        descriptions = [row['Summary'] for row in self.podatki]

        custom_stop_words = ['airport',
                             'aircraft',
                             'flight',
                             'crashed',
                             'flight',
                             'english',
                             'the',
                             'to',
                             'was',
                             'and',
                             'on',
                             'in',
                             'while',
                             'too',
                             'of',
                             'at',
                             'off', ]

        # Convert descriptions into TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=custom_stop_words)
        X = vectorizer.fit_transform(descriptions)

        # Apply PCA to reduce the dimensionality to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())

        # Apply K-means clustering
        num_clusters = 3  # You can adjust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # Get cluster centroids
        cluster_centers = kmeans.cluster_centers_

        # Get top keywords for each cluster
        top_keywords = []
        for i, centroid in enumerate(cluster_centers):
            top_keyword_indices = centroid.argsort()[-5:][::-1]  # Get indices of top 5 keywords
            top_keywords.append([vectorizer.get_feature_names_out()[index] for index in top_keyword_indices])

        # Collect example descriptions for each cluster
        examples_per_cluster = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            examples_per_cluster[label].append(descriptions[i])

        # Plot the clusters on a 2D scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.labels_, palette='viridis', legend='full', s=10)
        plt.title('Clustering of Crashes by Reason')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Add legend with top keywords for each cluster
        for i, keywords in enumerate(top_keywords):
            keywords_str = ', '.join(keywords)
            plt.text(X_pca[kmeans.labels_ == i, 0].mean(), X_pca[kmeans.labels_ == i, 1].mean(),
                     f'Cluster {i}: {keywords_str}', color='black', fontsize=10)

        for label, examples in examples_per_cluster.items():
            print(f"Cluster {label}:")
            for example in examples[:5]:  # Print up to 5 examples per cluster
                print(example)
            print()

        plt.tight_layout()
        plt.show()


#data = Data()
#data.get_geolocations()
