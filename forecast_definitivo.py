import streamlit as st

import pandas as pd
import numpy as np

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import json
from fbprophet.serialize import model_to_json, model_from_json
import holidays

import altair as alt
import base64
import itertools


st.set_page_config(page_title="Forecast App",
                   page_icon="🔮",
                   initial_sidebar_state="collapsed")


tabs = ["Application", "About"]
page = st.sidebar.radio("Tabs", tabs)


@st.cache(persist=False, suppress_st_warning=True, show_spinner=True, allow_output_mutation=True)
def load_csv(input_metric):
    df_input = None
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input_metric, sep=',', engine='python', encoding='utf-8',
                           parse_dates=True)
    return df_input.copy()


def prep_data(df):

    df_input = df.rename({date_col: "ds", metric_col: "y"},
                         errors='raise', axis=1)
    st.markdown(
        "The selected date column is now labeled as **ds** and the values columns as **y**")
    df_input = df_input[['ds', 'y']]
    df_input = df_input.sort_values(by='ds', ascending=True)
    df_input['ds'] = pd.to_datetime(df_input['ds'])
    df_input['y'] = df_input['y'].astype(float)
    return df_input.copy()


if page == "Application":

    st.title('Forecast application 🧙🏻')

    st.write(
        'This app enables you to generate time series forecast withouth any dependencies.')
    st.markdown(
        """The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")
    # caching.clear_cache()
    df = pd.DataFrame()

    st.subheader('1. Data loading 🏋️')
    st.write("Import a time series csv file.")
    with st.expander("Data format"):
        st.write("The dataset can contain multiple columns but you will need to select a column to be used as dates and a second column containing the metric you wish to forecast. The columns will be renamed as **ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric.")

    input = st.file_uploader('')

    if input:
        with st.spinner('Loading data..'):
            df = load_csv(input)

            st.write("Columns:")
            st.write(list(df.columns))
            columns = list(df.columns)

            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox(
                    "Select date column", index=0, options=columns, key="date")
            with col2:
                metric_col = st.selectbox(
                    "Select values column", index=1, options=columns, key="values")

            df = prep_data(df)
            output = 0

    if st.checkbox('Chart data', key='show'):
        with st.spinner('Plotting data..'):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df)

            with col2:
                st.write("Dataframe description:")
                st.write(df.describe())

        try:
            line_chart = alt.Chart(df).mark_line().encode(
                x='ds:T',
                y="y:Q", tooltip=['ds:T', 'y']).properties(title="Time series preview").interactive()
            st.altair_chart(line_chart, use_container_width=True)

        except:
            st.line_chart(df['y'], use_container_width=True, height=300)

    st.subheader("2. Parameters configuration 🛠️")

    with st.form("config"):

        with st.container():

            st.write('In this section you can modify the algorithm settings.')

            with st.expander("Horizon"):
                periods_input = st.number_input('Select how many future periods (days) to forecast.',
                                                min_value=1, max_value=366, value=90)

            with st.expander("Seasonality"):
                st.markdown(
                    """The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required. For more informations visit the [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)""")
                seasonality = st.radio(label='Seasonality', options=[
                                       'additive', 'multiplicative'])

            with st.expander("Trend components"):
                st.write("Add or remove components:")
                daily = st.checkbox("Daily")
                weekly = st.checkbox("Weekly")
                monthly = st.checkbox("Monthly")
                yearly = st.checkbox("Yearly")

            # with st.expander("Covid modeling"):
            #     st.write("Add covid impact?")

            #     covid = st.checkbox("Covid")
            #     primera_ola = None
            #     segunda_ola = None
            #     if covid:
            #         primera_ola = pd.DataFrame({
            #             'holiday': 'primera_ola',
            #             'ds': pd.date_range(pd.to_datetime('2020-03-14'), periods=79).tolist(),
            #             'lower_window': 0,
            #             'upper_window': 0,
            #         })
            #         segunda_ola = pd.DataFrame({
            #             'holiday': 'segunda_ola',
            #             'ds': pd.date_range(pd.to_datetime('2020-08-04'), periods=280).tolist(),
            #             'lower_window': 0,
            #             'upper_window': 0,
            #         })
            #         covid_dates = primera_ola.append(segunda_ola)

            with st.expander("Growth model"):
                
                st.write('Prophet uses by default a linear growth model.')
                st.markdown(
                    """For more information check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)""")
                st.write('Configure saturation (for logistic growth only )')
                
                growth = st.radio(label='Growth model',options=['linear',"logistic"]) 

            if growth == 'linear':
                growth_settings= {
                            'cap':1,
                            'floor':0
                        }
                cap=1
                floor=1
                df['cap']=1
                df['floor']=0

            if growth == 'logistic':
                st.info('Configure saturation')

                cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
                floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
                if floor > cap:
                    st.error('Invalid settings. Cap must be higher then floor.')
                    growth_settings={}

                if floor == cap:
                    st.warning('Cap must be higher than floor')
                else:
                    growth_settings = {
                        'cap':cap,
                        'floor':floor
                        }
                    df['cap']=cap
                    df['floor']=floor
                
                # growth = st.radio(label='Growth model', options=[
                #                   'linear', "logistic"])
                # saturation = st.slider(label="+/- Growth factor (%)",
                #                        max_value=1.0,
                #                        min_value=0.01)

                #growth = st.radio(label='Growth model',options=['Linear',"Logistic"])
                # if growth == 'linear':

                #     growth_settings= {
                #                'cap':1,
                #                'floor':0
                #          }
                #     cap=1
                #     floor=1
                #     df['cap']=1
                #     df['floor']=0

                # if growth == "logistic":

                #     st.write('Configure saturation')
                #     saturation = st.slider(label="+/- Growth factor (%)",
                #                         max_value=1.0,
                #                         min_value=0.00)

                #     if input :

                #         last_year_df = df.tail(365).copy()
                #         quantil_cap = last_year_df.y.quantile(.75)
                #         quantil_floor = last_year_df.y.quantile(.25)

                # cap=( 1 + saturation) * quantil_cap
                # floor=( 1 - saturation) * quantil_floor
                # st.markdown(f"Cap: {cap} \n Floor: {floor}")
                # growth_settings = {
                #     'cap':cap,
                #     'floor':floor}

                # df['cap']=cap
                # df['floor']=floor

            with st.expander('Holidays'):

                countries = ['Country name', 'Italy', 'Spain',
                             'United States', 'France', 'Germany', 'Ukraine']

                with st.container():
                    years = [2022]
                    selected_country = st.selectbox(
                        label="Select country", options=countries)

                    if selected_country == 'Italy':
                        for date, name in sorted(holidays.IT(years=years).items()):
                            st.write(date, name)

                    if selected_country == 'Spain':

                        for date, name in sorted(holidays.ES(years=years).items()):
                            st.write(date, name)

                    if selected_country == 'United States':

                        for date, name in sorted(holidays.US(years=years).items()):
                            st.write(date, name)

                    if selected_country == 'France':

                        for date, name in sorted(holidays.FR(years=years).items()):
                            st.write(date, name)

                    if selected_country == 'Germany':

                        for date, name in sorted(holidays.DE(years=years).items()):
                            st.write(date, name)

                    if selected_country == 'Ukraine':

                        for date, name in sorted(holidays.UKR(years=years).items()):
                            st.write(date, name)

                    else:
                        holidays = False

                    holidays = st.checkbox('Add country holidays to the model')

            with st.expander('Hyperparameters'):
                st.write(
                    'In this section it is possible to tune the scaling coefficients.')

                seasonality_scale_values = [0.1, 1.0, 5.0]
                changepoint_scale_values = [ 0.1, 0.5, 1.0]

                st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
                changepoint_scale = st.select_slider(
                    label='Changepoint prior scale', options=changepoint_scale_values)

                st.write(
                    "The seasonality change point controls the flexibility of the seasonality.")
                seasonality_scale = st.select_slider(
                    label='Seasonality prior scale', options=seasonality_scale_values)

                st.markdown(
                    """For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")
                # falta el seasonality holydays scale
                # y que el slider seas continuos no discretos
        submitted = st.form_submit_button("Submit")

        if submitted:
            
            # if growth == 'linear':
            #     # growth_settings= {
            #     #             'cap':1,
            #     #             'floor':0}
            #     cap = 1
            #     floor = 1
            #     df['cap'] = 1
            #     df['floor'] = 0

            # if growth == 'logistic':

            #     last_year_df = df.tail(365).copy()
            #     quantil_cap = last_year_df.y.quantile(.75)
            #     quantil_floor = last_year_df.y.quantile(.25)

            #     cap = (1 + saturation) * quantil_cap
            #     floor = (1 - saturation) * quantil_floor
            #     #st.markdown(f"Cap: {cap} \n Floor: {floor}")
            #     # growth_settings = {
            #     #     'cap':cap,
            #     #     'floor':floor}
            #     df['cap'] = cap
            #     df['floor'] = floor

            st.markdown(f""" Model Configuration: \n
            Horizon: {periods_input}  days    \n
            Seasonality: {seasonality}  \n
            Trend components: {daily};{weekly};{monthly};{yearly} \n
            Growth: {growth} 
            Holidays: {selected_country} \n
            Hyperparameters: changepoints{changepoint_scale}, seasonality {seasonality_scale}
            """)
            st.success("Configuration submitted")
            st.write(df.head())

    # growth with radio button
    # with st.expander("Growth model"):
    #         st.write('Prophet uses by default a linear growth model.')
    #         st.markdown("""For more information check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)""")

    #         growth = st.radio(label='Growth model',options=['Linear',"Logistic"])

    #         if growth == 'Linear':

    #             growth_settings= {
    #                         'cap':1,
    #                         'floor':0}
    #             cap=1
    #             floor=1
    #             df['cap']=1
    #             df['floor']=0

    #         if growth == 'Logistic':

            # if input :
            #     st.write('Configure saturation')

            #     last_year_df = df.tail(365).copy()
            #     quantil_cap = last_year_df.y.quantile(.75)
            #     quantil_floor = last_year_df.y.quantile(.25)

            #     saturation = st.slider(label="+/- Growth factor (%)",
            #                             max_value=1.0,
            #                             min_value=0.00)
            #     st.write(saturation)

            #     cap=( 1+ saturation) * quantil_cap
            #     floor=( 1 - saturation) * quantil_floor

            #     st.markdown(f"Cap: {cap} \n Floor: {floor}")

            #     growth_settings = {
            #         'cap':cap,
            #         'floor':floor}

            #     df['cap']=cap
            #     df['floor']=floor

    st.write("Below you can upload further regressors to the forecast")
    with st.expander("Regressors"):

        regressor_input = st.file_uploader(
            'Upload time series of values that have an impact on the time series you are predicting.')

        if regressor_input:

            metric_df = load_csv(regressor_input)

            st.write("Columns:")
            st.write(list(metric_df.columns))
            columns_2 = list(metric_df.columns)

            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox(
                    "Select date column", index=0, options=columns_2, key="date_2")
            with col2:
                metric_col = st.selectbox(
                    "Select values column", index=1, options=columns_2, key="values_2")

            metric_df = prep_data(metric_df)
            output = 0

    with st.container():
        st.subheader("3. Forecast 🔮")
        st.write("Fit the model on the data and generate future prediction.")
        st.write("Load a time series to activate.")

        if input:
           # if covid:
            if st.checkbox("Initialize model (Fit)", key="fit"):
                # if len(growth_settings)==2:
                m = Prophet(seasonality_mode=seasonality,
                            daily_seasonality=daily,
                            weekly_seasonality=weekly,
                            yearly_seasonality=yearly,
                            growth=growth,
                            changepoint_prior_scale=changepoint_scale,
                            seasonality_prior_scale=seasonality_scale,
                           # holidays=covid_dates
                            )
                if holidays:
                    m.add_country_holidays(country_name=selected_country)

                if monthly:
                    m.add_seasonality(
                        name='monthly', period=30.4375, fourier_order=5)
                if regressor_input:
                    m.add_regressor(metric_name)
                    df = pd.merge(df, metric_df, how="left", on="ds")
                with st.spinner('Fitting the model..'):

                    m = m.fit(df)

                    future = m.make_future_dataframe(
                        periods=periods_input, freq='D')

                    future['cap'] = cap
                    future['floor'] = floor
                    st.write(future.head())
                    st.write(
                        "The model will produce forecast up to ", future['ds'].max())
                    st.success('Model fitted successfully')

                if regressor_input:
                    future = pd.merge(future, metric_df,
                                        how="left", on="ds")

                # else:
                #     st.warning('Invalid configuration')

                # if st.checkbox("Initialize model (Fit)",key="fit"):
                #     #if len(growth_settings)==2:
                #     m = Prophet(seasonality_mode=seasonality,
                #                 daily_seasonality=daily,
                #                 weekly_seasonality=weekly,
                #                 yearly_seasonality=yearly,
                #                 growth=growth,
                #                 changepoint_prior_scale=changepoint_scale,
                #                 seasonality_prior_scale= seasonality_scale,
                #                 )
                #     if holidays:
                #         m.add_country_holidays(country_name=selected_country)

                #     if monthly:
                #         m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)

                #     with st.spinner('Fitting the model..'):

                #         m = Prophet(growth=_model,
                #                     seasonality_mode=_modality,
                #                     holidays_prior_scale=x2,
                #                     changepoint_range=x1,
                #                     changepoint_prior_scale=x0,
                #                     holidays=holidays, daily_seasonality=False)

                #         m = m.fit(df)
                #         future = m.make_future_dataframe(periods=periods_input,freq='D')
                #         future['cap']=cap
                #         future['floor']=floor
                #         st.write("The model will produce forecast up to ", future['ds'].max())
                #         st.success('Model fitted successfully')

                # else:
                    # .warning('Invalid configuration')

            if st.checkbox("Generate forecast (Predict)", key="predict"):
                try:
                    with st.spinner("Forecasting.."):

                        forecast = m.predict(future)
                        st.success('Prediction generated successfully')
                        st.dataframe(forecast)
                        fig1 = m.plot(forecast)
                        st.write(fig1)
                        output = 1

                        if growth == 'linear':
                            fig2 = m.plot(forecast)
                            a = add_changepoints_to_plot(
                                fig2.gca(), m, forecast)
                            st.write(fig2)
                            output = 1
                except:
                    st.warning("You need to train the model first.. ")

            if output == 1:
                if st.checkbox('Show components'):
                    try:
                        with st.spinner("Loading.."):
                            fig3 = m.plot_components(forecast)
                            st.write(fig3)
                    except:
                        st.warning("Requires forecast generation")
                    
            
            if input:
                if output == 1:
                    
                    # with st.expander('Download forecast'):
                    @st.cache
                    def convert_df(df):
                         
                        
                        # IMPORTANT: Cache the conversion to prevent computation on every rerun
                        return df.to_csv().encode('utf-8')

                    csv = convert_df( pd.DataFrame(
                                forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']]))
                    
                   
                    st.download_button(label="Download data as CSV",
                                        data=csv,
                                        file_name='forecast.csv',
                                        mime='text/csv')
                        
                        
                        
                        # if st.button('Export forecast (.csv)'):
                        #     with st.spinner("Exporting.."):

                        #         export_forecast = pd.DataFrame(
                        #             forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']]).to_csv()
                        #         b64 = base64.b64encode(
                        #             export_forecast.encode()).decode()
                        #         href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right click  > save as  **forecast.csv**)'
                        #         st.markdown(href, unsafe_allow_html=True)

                    


        st.subheader('4. Model validation 🧪')
        st.write(
            "In this section it is possible to do cross-validation of the model.")
        with st.expander("Cross validation"):
            st.markdown("""The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
            st.write(
                "Training data (initial): The amount of data set aside for training. The parameter is in the API called initial.")
            st.write("Horizon: The data set aside for validation.")
            st.write(
                "Cutoff (period): a forecast is made for every observed point between cutoff and cutoff + horizon.""")

        # with st.expander("Cross validation"):
            
                                     
            # initial = st.number_input(
            #     value=365, label="initial", min_value=30, max_value=1096)
            # initial = str(initial) + " days"

            # period = st.number_input(
            #     value=90, label="period", min_value=1, max_value=365)
            # period = str(period) + " days"

            # horizon = st.number_input(
            #     value=90, label="horizon", min_value=30, max_value=366)
            # horizon = str(horizon) + " days"

            initial = st.number_input(value= 120,label="initial",min_value=30,max_value=1096)
            initial = str(initial) + " days"

            period = st.number_input(value= 60,label="period",min_value=1,max_value=365)
            period = str(period) + " days"

            horizon = st.number_input(value= 90, label="horizon",min_value=30,max_value=366)
            horizon = str(horizon) + " days"

            st.write(f"Here we do cross-validation to assess prediction performance on a horizon of **{horizon}** , starting with **{initial}**  of training data in the first cutoff and then making predictions every **{period}**.")
            st.markdown("""For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")
        

        # with st.expander("Metrics"):

            if input:
                if output == 1:
                    if st.checkbox('Calculate metrics'):
                        with st.spinner("Cross validating.."):
                            try:
                                df_cv = cross_validation(m, initial=initial,
                                                         period=period,
                                                         horizon=horizon,
                                                         parallel="processes")
                                df_p = performance_metrics(df_cv)
                                st.write(df_p)
                                metrics = 1
                                
                                if metrics == 1:
                                    st.markdown('**Metrics definition**')
                                    st.write("Mse: mean absolute error")
                                    st.write("Mae: Mean average error")
                                    st.write("Mape: Mean average percentage error")
                                    st.write("Mse: mean absolute error")
                                    st.write("Mdape: Median average percentage error")

                                    metrics = ['Choose a metric', 'mse', 'rmse',
                                            'mae', 'mape', 'mdape', 'coverage']
                                    selected_metric = st.selectbox(
                                        "Select metric to plot", options=metrics)
                                    if selected_metric != metrics[0]:
                                        fig4 = plot_cross_validation_metric(
                                            df_cv, metric=selected_metric)
                                        st.write(fig4)
                                    
                            except:
                                st.write("Invalid configuration, try other periods")

            else:
                st.write("Create a forecast to see metrics")

        st.subheader('5. Hyperparameter Tuning 🧲')
        st.write(
            "In this section it is possible to find the best combination of hyperparamenters.")
        st.markdown(
            """For more informations visit the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)""")

        param_grid = {
            'changepoint_prior_scale': [0.1, 1.0, 5.0],
            'seasonality_prior_scale': [ 0.1, 0.5, 1.0],
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v))
                      for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        if input:
            if output == 1:

                if st.button("Optimize hyperparameters"):

                    with st.spinner("Finding best combination. Please wait.."):

                        try:
                            # Use cross validation to evaluate all parameters
                            for params in all_params:
                                # Fit model with given params
                                m = Prophet(**params).fit(df)
                                df_cv = cross_validation(m, initial=initial,
                                                         period=period,
                                                         horizon=horizon,
                                                         parallel="processes")
                                df_p = performance_metrics(
                                    df_cv, rolling_window=1)
                                rmses.append(df_p['rmse'].values[0])
                        except:
                            for params in all_params:
                                # Fit model with given params
                                m = Prophet(**params).fit(df)
                                df_cv = cross_validation(m, initial=initial,
                                                         period=period,
                                                         horizon=horizon,
                                                         parallel="threads")
                                df_p = performance_metrics(
                                    df_cv, rolling_window=1)
                                rmses.append(df_p['rmse'].values[0])

                    # Find the best parameters
                    tuning_results = pd.DataFrame(all_params)
                    tuning_results['rmse'] = rmses
                    st.write(tuning_results)

                    best_params = all_params[np.argmin(rmses)]

                    st.write('The best parameter combination is:')
                    st.write(best_params)
                    #st.write(f"Changepoint prior scale:  {best_params[0]} ")
                    #st.write(f"Seasonality prior scale: {best_params[1]}  ")
                    st.write(
                        " You may repeat the process using these parameters in the configuration section 2")

            else:
                st.write("Create a model to optimize")

        st.subheader('6. Export results ✨')

        st.write(
            "Finally you can export your result forecast, model configuration and evaluation metrics.")

        if input:
            if output == 1:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(label="Download forecast",
                                        data=csv,
                                        file_name='Forecast.csv',
                                        mime='text/csv')
                    # if st.button('Export Forecast (.csv)'):
                    #     with st.spinner("Exporting.."):

                    #         export_forecast = pd.DataFrame(
                    #             forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']]).to_csv()
                    #         b64 = base64.b64encode(
                    #             export_forecast.encode()).decode()
                    #         href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right click > save as **forecast.csv**)'
                    #         st.markdown(href, unsafe_allow_html=True)

                with col2:
                    if st.button("Export model metrics (.csv)"):
                        try:
                            df_p = df_p.to_csv(decimal=',')
                            b64 = base64.b64encode(df_p.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **metrics.csv**)'
                            st.markdown(href, unsafe_allow_html=True)
                        except:
                            st.write("No metrics to export")

                with col3:
                    if st.button('Export Model Configuration (.json)'):
                        with st.spinner("Exporting.."):
                            with open('serialized_model.json', 'w') as fout:
                                json.dump(model_to_json(m), fout)

            

            else:
                st.write("Generate a forecast to download.")


if page == "About":
    st.image("prophet.png")
    st.header("About")
    st.markdown(
        "Official documentation of **[Facebook Prophet](https://facebook.github.io/prophet/)**")
    st.markdown(
        "Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.write("Author:")
    st.markdown(
        """ **[Giancarlo Di Donato](https://www.linkedin.com/in/giancarlodidonato/)**""")
    st.markdown(
        """**[Source code](https://github.com/giandata/forecast-app)**""")

    st.write("Created on 27/02/2021")
    st.write("Last updated: **17/02/2022**")
