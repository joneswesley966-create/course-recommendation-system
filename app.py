import streamlit as st
from model import recommend_courses

st.title("🎓 course recommendation system")

st.write("enter User ID to get course recommendations")

user_id = st.text_input("UserID (Example: U00001)")

if st.button("Recommend"):
    if user_id:
        try:
            recs = recommend_course(user_id)
            st.success("Recommend courses:")
            for course in recs: st.write(f"course ID:{course}")
        except:
            st.error("Users not found!")
        else:
            st.warning("please enter a User ID")

recs = recommend_courses(user_id)

if isinstance(recs, str):
    st.error(recs)
else:
    st.table(recs)