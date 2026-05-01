import sys
import os
import time
import traceback
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_data
from models.lstm_pytorch import run_lstm as run_lstm_model
from models.kmeans_model import label_clusters
from agents.inventory_agent import inventory_agent_summary
from agents.customer_agent import customer_agent_summary
from agents.profit_agent import profit_agent_summary
from agents.recommendation_agent import recommendation_agent_summary


def get_data_context(df):
    context = {}
    try:
        context['total_products'] = df['Product_Name'].nunique()
        context['total_customers'] = df['Customer_ID'].nunique()
        context['date_min'] = df['Date'].min().date()
        context['date_max'] = df['Date'].max().date()
        total_p = df["Profit"].sum()
        context['total_profit'] = total_p
        context['profit_status'] = 'Profitable ' if total_p > 0 else 'In Loss '
        context['profit_badge'] = 'badge-profit' if total_p > 0 else 'badge-loss'
    except Exception:
        pass
    return context


def sales_inventory(request):
    try:
        df = preprocess_data()
    except Exception as e:
        return render(request, 'dashboard/error.html', {'error': f"Failed to load data: {e}"})

    context = get_data_context(df)
    refresh = request.GET.get('refresh')
    if refresh:
        request.session.flush()

    try:
        profit_data = profit_agent_summary(df)
        inventory = inventory_agent_summary(df)

        context['total_profit_display'] = f"${profit_data['total_profit']:,.0f}"
        context['top_products_count'] = len(profit_data["top_products"])
        context['low_stock_count'] = len(inventory["low_stock_products"])
        context['festival_count'] = len(inventory["festival_top_products"])

        sales = df.groupby("Date")["Quantity_Sold"].sum().reset_index()
        context['sales_dates'] = sales["Date"].dt.strftime("%Y-%m-%d").tolist()
        context['sales_values'] = sales["Quantity_Sold"].tolist()

        top = profit_data["top_products"].reset_index()
        top.columns = ["Product", "Profit"]
        context['top_products'] = top.to_dict('records')

        loss = profit_data.get("loss_products")
        context['has_loss'] = loss is not None and not loss.empty
        if context['has_loss']:
            loss_df = loss.reset_index().rename(columns={"Product_Name": "Product", "Profit": "Loss"})
            context['loss_products'] = loss_df.to_dict('records')

        low_stock = inventory["low_stock_products"]
        restock = inventory["restock_suggestions"]
        if not low_stock.empty:
            low_df = low_stock.reset_index()
            low_df.columns = ["Product", "Current Stock"]
            low_df["Restock Qty"] = low_df["Product"].map(restock).fillna(0).astype(int)
            context['low_stock_items'] = low_df.to_dict('records')
            context['has_low_stock'] = True
        else:
            context['has_low_stock'] = False

        fest = inventory["festival_top_products"].reset_index()
        fest.columns = ["Product", "Units Sold"]
        context['festival_products'] = fest.to_dict('records')
        context['inventory_insights'] = inventory["insights"]

    except Exception as e:
        context['error'] = f"KPI Row failed: {e}"

    return render(request, 'dashboard/sales_inventory.html', context)


def customers_recommendations(request):
    try:
        df = preprocess_data()
    except Exception as e:
        return render(request, 'dashboard/error.html', {'error': f"Failed to load data: {e}"})

    context = get_data_context(df)

    try:
        clusters = label_clusters(df)

        segment_data = []
        for seg in ["High Value", "Medium Value", "Low Value"]:
            count = len(clusters[clusters["Segment"] == seg])
            segment_data.append({"name": seg, "count": count})
        context['segment_data'] = segment_data

        seg_counts = clusters["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        context['segment_labels'] = seg_counts["Segment"].tolist()
        context['segment_counts'] = seg_counts["Count"].tolist()

        display_cols = [c for c in ["Customer_ID", "Quantity_Sold", "Profit", "Segment"] if c in clusters.columns]
        customer_df = clusters[display_cols].head(20)
        context['customer_columns'] = display_cols
        context['customer_rows'] = customer_df.values.tolist()

        customer = customer_agent_summary(df)
        hv = customer["high_value_customers"]
        lv = customer["low_value_customers"]
        context['high_value_columns'] = hv.columns.tolist()
        context['high_value_rows'] = hv.values.tolist()
        context['low_value_columns'] = lv.columns.tolist()
        context['low_value_rows'] = lv.values.tolist()
        context['customer_insights'] = customer["insights"]

        rec = recommendation_agent_summary()
        top_rules = rec["top_rules"]
        if not top_rules.empty:
            context['has_rules'] = True
            rules_dict = top_rules.to_dict('records')
            for r in rules_dict:
                r['antecedents'] = str(r['antecedents'])
                r['consequents'] = str(r['consequents'])
            context['top_rules'] = rules_dict
        else:
            context['has_rules'] = False

        context['rec_insights'] = rec["insights"]

    except Exception as e:
        context['error'] = f"Customer Analysis failed: {e}"

    return render(request, 'dashboard/customers_recommendations.html', context)


def forecasting(request):
    try:
        df = preprocess_data()
    except Exception as e:
        return render(request, 'dashboard/error.html', {'error': f"Failed to load data: {e}"})

    context = get_data_context(df)

    try:
        sales = df.groupby("Date")["Quantity_Sold"].sum().reset_index()
        recent = sales.tail(30)
        context['recent_dates'] = recent["Date"].dt.strftime("%Y-%m-%d").tolist()
        context['recent_values'] = recent["Quantity_Sold"].tolist()

        all_sales = df.groupby("Date")["Quantity_Sold"].sum()
        context['avg_7'] = f"{all_sales.tail(7).mean():.0f}"
        context['avg_30'] = f"{all_sales.tail(30).mean():.0f}"

        if "lstm_result" in request.session:
            pred = request.session["lstm_result"]
            elapsed = request.session.get("lstm_time", 0)
            context['prediction'] = f"{pred:.0f}"
            context['prediction_time'] = f"{elapsed:.1f}"
            avg = all_sales.mean()
            diff = pred - avg
            direction = "above" if diff > 0 else "below"
            context['pred_diff'] = f"{abs(diff):.0f}"
            context['pred_direction'] = direction
            context['daily_avg'] = f"{avg:.0f}"
            context['has_prediction'] = True
        else:
            context['has_prediction'] = False

    except Exception as e:
        context['error'] = f"Forecasting failed: {e}"

    return render(request, 'dashboard/forecasting.html', context)


@csrf_exempt
def run_lstm(request):
    if request.method == 'POST':
        try:
            df = preprocess_data()
            start = time.time()
            prediction = run_lstm_model(df)
            elapsed = time.time() - start

            request.session["lstm_result"] = prediction
            request.session["lstm_time"] = elapsed

            return JsonResponse({
                'success': True,
                'prediction': f"{prediction:.0f}",
                'time': f"{elapsed:.1f}"
            })
        except Exception as e:
            error_trace = traceback.format_exc()
            print("LSTM Error:", error_trace)
            return JsonResponse({'success': False, 'error': str(e), 'traceback': error_trace})

    return JsonResponse({'success': False, 'error': 'Invalid method'})
